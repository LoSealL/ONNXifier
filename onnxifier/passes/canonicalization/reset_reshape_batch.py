"""
Copyright (C) 2025 The ONNXIFIER Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set

import networkx as nx
import numpy as np
from onnx import NodeProto
from onnx.numpy_helper import to_array

from ... import logger
from ...graph import OnnxGraph
from .. import PASSES, Rewriter
from ..pattern import SingleNodePattern
from ..utils import evaluate_on_node, make_constant


def _ensure_simple_batch(node_name, indices: Set[int]):
    if len(indices) > 1:
        raise NotImplementedError(
            f"The input of operator {node_name} has complex batch definition."
        )


def _get_dim_after_transpose(node_pb: NodeProto, index: int) -> int:
    perm: List[int] = []
    for t in node_pb.attribute:
        if t.name == "perm":
            perm = list(t.ints)
    assert perm
    return perm.index(index)


def _get_dim_after_reshape(graph: OnnxGraph, node_pb: NodeProto, index: int) -> int:
    # To guess the dimension merge and split, it not supported on symbolic shape.
    input_shape = graph.static_tensor_shape(node_pb.input[0])
    output_shape = graph.static_tensor_shape(node_pb.output[0])
    batch = input_shape[index]
    in_s1, in_s2 = input_shape[:index], input_shape[index + 1 :]
    if batch in output_shape and output_shape.count(batch) == 1:
        out_ind = output_shape.index(batch)
        out_s1, out_s2 = output_shape[:out_ind], output_shape[out_ind + 1 :]
        if (np.prod(in_s1) == np.prod(out_s1)) and (np.prod(in_s2) == np.prod(out_s2)):
            return out_ind
        else:
            raise ValueError(
                f"Can't trace complex reshape {input_shape} -> {output_shape}"
            )
    # Search for (i, j, k) that index in [i, j) and
    # prod(input_shape[i:j]) == output_shape[k]
    for i, j in itertools.product(
        range(index - 1, -1, -1), range(index, len(input_shape))
    ):
        for k, _ in enumerate(output_shape):
            if np.prod(input_shape[i : j + 1]) == output_shape[k]:
                logger.debug(f"Found batch in output shape {output_shape}[{k}]")
                return k

    raise NotImplementedError(f"Complex reshape {input_shape} -> {output_shape}")


def _canonicalize_neg_axes(axes, rank: int) -> List[int]:
    if axes.ndim == 0:
        axes = axes[None]
    axes = list(axes)
    for i, axis in enumerate(axes):
        if axis < 0:
            axes[i] = rank + axis
    return axes


def _get_dim_after_squeeze(graph: OnnxGraph, node_pb: NodeProto, index: int) -> int:
    axes: Optional[Sequence[int] | np.ndarray] = []
    if node_pb.input[1] in graph.initializers:
        axes = to_array(graph.initializers[node_pb.input[1]])
    else:
        axes_node = graph.onnx_predecessors(node_pb)[-1]
        axes = evaluate_on_node(graph, axes_node)
    if axes is None:
        raise ValueError(f'Can\'t trace squeeze "{node_pb.name}" with unknown axes')
    rank = len(graph.tensor_shape(node_pb.input[0]))
    axes = _canonicalize_neg_axes(axes, rank)
    if index in axes:
        raise ValueError(f'Batch is squeezed out in "{node_pb.name}"')
    for i in sorted(axes):
        if i < index:
            index -= 1
    assert index >= 0
    return index


def _get_dim_after_unsqueeze(graph: OnnxGraph, node_pb: NodeProto, index: int) -> int:
    axes: Optional[Sequence[int] | np.ndarray] = []
    if node_pb.input[1] in graph.initializers:
        axes = to_array(graph.initializers[node_pb.input[1]])
    else:
        axes_node = graph.onnx_predecessors(node_pb)[-1]
        axes = evaluate_on_node(graph, axes_node)
    if axes is None:
        raise ValueError(f'Can\'t trace unsqueeze "{node_pb.name}" with unknown axes')
    rank = len(graph.tensor_shape(node_pb.input[0]))
    # note unsqueeze(x, -1) is equivalent to unsqueeze(x, x.ndim)
    axes = _canonicalize_neg_axes(axes, rank + 1)
    for i in sorted(axes):
        if i <= index:
            index += 1
    return index


def _get_dim_after_gather(graph: OnnxGraph, node_pb: NodeProto, index: int) -> int:
    axis = 0
    for t in node_pb.attribute:
        if t.name == "axis":
            axis = t.i
    rank = len(graph.tensor_shape(node_pb.input[1]))
    if axis == index and rank == 0:
        raise ValueError(f'Gather "{node_pb.name}" eliminates the batch dimension')
    return index


def _dispatch_shuffle_op(graph: OnnxGraph, node_pb: NodeProto, index: int) -> int:
    if node_pb.op_type == "Transpose":
        index = _get_dim_after_transpose(node_pb, index)
    elif node_pb.op_type == "Reshape":
        index = _get_dim_after_reshape(graph, node_pb, index)
    elif node_pb.op_type == "Squeeze":
        index = _get_dim_after_squeeze(graph, node_pb, index)
    elif node_pb.op_type == "Unsqueeze":
        index = _get_dim_after_unsqueeze(graph, node_pb, index)
    elif node_pb.op_type == "Gather":
        index = _get_dim_after_gather(graph, node_pb, index)
    return index


def trace_batch_dimension(
    graph: OnnxGraph, batch_index: Optional[Dict[str, int]] = None
):
    """trace the batch dimension of the graph.

    Args:
        graph (OnnxGraph): The input graph.
        batch_index (Dict[str, int]): The argument to specify the index of the batch
            dimension of each inputs. Defaults to all 0.
    """

    if batch_index is None:
        batch_index = {i: 0 for i in graph.inputs}
    for name, idx in batch_index.items():
        logger.debug(f'batch dim of "{name}" is {idx}')

    # tracking the dimension index of the inputs,
    # change only after {transpose, reshape}.
    input_index_tracing: Dict[str, Set[int]] = defaultdict(set)
    output_index_tracing: Dict[str, Set[int]] = defaultdict(set)
    for node_name in nx.topological_sort(graph):
        node_pb: NodeProto = graph.nodes[node_name]["pb"]
        if graph.nodes[node_name]["has_input"]:
            indices: Set[int] = set()
            for input_name in node_pb.input:
                if input_name in batch_index:
                    indices.add(batch_index[input_name])
            assert indices
            _ensure_simple_batch(node_name, indices)
            index = indices.pop()
            index = _dispatch_shuffle_op(graph, node_pb, index)
            output_index_tracing[node_name].add(index)
            for i in graph.successors(node_name):
                input_index_tracing[i].add(index)
        elif node_name in input_index_tracing:
            indices = input_index_tracing[node_name].copy()
            _ensure_simple_batch(node_name, indices)
            index = indices.pop()
            index = _dispatch_shuffle_op(graph, node_pb, index)
            output_index_tracing[node_name].add(index)
            for i in graph.successors(node_name):
                input_index_tracing[i].add(index)

    for name, indices in output_index_tracing.items():
        _ensure_simple_batch(name, indices)
        logger.trace(f"{name}: batch dim = {indices}")

    return output_index_tracing


@PASSES.register("reset_reshape_batch")
class ResetReshapeBatchRewriter(Rewriter):
    """Reset the batch dimension of Reshape to -1.

    Batch dimension is tracked by :func:`trace_batch_dimension` function.
    """

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Reshape"))
        self.index_tracing: Dict[str, Set[int]] = {}

    def match_and_rewrite(self, graph: OnnxGraph, *args, **kwargs) -> OnnxGraph:
        self.index_tracing = trace_batch_dimension(graph, kwargs.get("batch_index"))
        return super().match_and_rewrite(graph, *args, **kwargs)

    def rewrite(self, graph, nodes, *args, **kwargs):
        node = nodes[0]
        shape = self.get_value(node.input[1])
        if shape is None:
            logger.debug(f'target shape of "{node.name}" is dynamic')
            return

        if node.name in self.index_tracing:
            indices = self.index_tracing[node.name].copy()
            if len(indices) != 1:
                return
            index = indices.pop()
            # fill exist -1
            output_shape = graph.tensor_shape(node.output[0])
            shape = list(shape)
            for i, s in enumerate(shape):
                if s == -1:
                    shape[i] = output_shape[i]
            # reset batch dim to -1
            shape[index] = -1
            shape_cst = make_constant(f"{node.name}/shape", np.array(shape, np.int64))
            node.input[1] = shape_cst.output[0]
            self += shape_cst
