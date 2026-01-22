"""
Copyright (C) 2024-2025 The ONNXIFIER Authors.

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

from typing import List, Optional, Sequence

import networkx as nx
import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph, logger
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register("reorder_transpose_to_downstream")
class ReorderTransposeToDownstreamRewriter(Rewriter):
    """Reorder Transpose operator to place it after the element-wise operations.

    Before:

        Transpose -> Add -> Relu -> Mul

    After:

        Add -> Relu -> Mul -> Transpose
    """

    _binary_ops = {"Add", "Sub", "Mul", "Div", "Pow", "Mod", "Exp", "Log"}
    _unary_ops = {"Relu", "LeakyRelu", "PRelu", "Sigmoid", "Tanh", "Erf"}
    _reaxis_ops = {"Concat", "Slice", "Split", "Softmax", "Tile", "Pad"}
    _axis_change_ops = {"Squeeze", "Unsqueeze"}
    _reduce_ops = {"ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceL2"}

    def __init__(self):
        super().__init__(SingleNodePattern("Transpose"))
        self._supported_ops = (
            self._binary_ops.union(self._unary_ops)
            .union(self._reaxis_ops)
            .union(self._axis_change_ops)
            .union(self._reduce_ops)
        )

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        trans_node = nodes[0]
        allowed_nodes = set()
        for i in nx.descendants(graph, trans_node.name):
            if graph.nodes[i]["pb"].op_type in self._supported_ops:
                allowed_nodes.add(i)
        h = graph.onnx_subgraph(allowed_nodes | {trans_node.name})
        allowed_and_connected = set()
        for i in allowed_nodes:
            if nx.has_path(h, trans_node.name, i):
                allowed_and_connected.add(self.graph.nodes[i]["pb"])
        logger.debug(f"# of nodes to reorder: {len(allowed_and_connected)}")
        logger.trace(f"{allowed_and_connected}")
        if not allowed_and_connected:
            return
        h = graph.onnx_subgraph(allowed_and_connected | {trans_node.name})
        if len(h.output) > 1:
            # can't reorder to diverged outputs
            logger.debug(f"Can't reorder to diverged outputs: {len(h.output)}")
            return

        perm = self.get_attribute(trans_node, "perm")
        assert isinstance(perm, list)
        for node in allowed_and_connected:
            if node.op_type in self._unary_ops:
                continue
            elif node.op_type in self._binary_ops:
                tracing = self._get_path(graph, trans_node.name, node.name)
                if node.input[0] in tracing and node.input[1] in tracing:
                    # both inputs are traced, skip
                    continue
                else:
                    self._rewrite_binary_op(
                        graph,
                        node,
                        node.input[1] if node.input[0] in tracing else node.input[0],
                        perm,  # type: ignore
                    )
            elif node.op_type in self._reaxis_ops:
                self._rewrite_reaxis_op(
                    graph,
                    node,
                    perm,  # type: ignore
                )
            else:
                raise NotImplementedError(f"Unsupported op {node.op_type}")

        # reorder transpose node
        for node in graph.onnx_successors(trans_node):
            for i, name in enumerate(node.input):
                if name in trans_node.output:
                    node.input[i] = trans_node.input[0]
        self -= trans_node
        new_trans_node = make_node(
            "Transpose",
            [h.output[0].name],
            trans_node.output,
            name=f"{trans_node.name}/reorder",
            perm=perm,
        )
        self += new_trans_node
        # the last node in subgraph "h"
        out_node = graph.nodes[h._out_to_node[h.output[0].name]]["pb"]
        for node in graph.onnx_successors(out_node):
            for i, name in enumerate(node.input):
                if name in out_node.output:
                    node.input[i] = new_trans_node.output[0]

    def _get_path(self, graph, src, dst):
        traced_path = []
        for path in nx.all_simple_paths(graph, src, dst):
            curr_output = []
            for i in path:
                node = graph.nodes[i]["pb"]
                for j in node.input:
                    if j in curr_output:
                        traced_path.append(j)
                curr_output.extend(node.output)
        return set(traced_path)

    def _rewrite_binary_op(
        self, graph: OnnxGraph, node: NodeProto, another_input: str, perm: List[int]
    ):
        inv_perm = [perm.index(i) for i in range(len(perm))]
        shape = graph.tensor_shape(another_input)
        if len(shape) <= 1:
            return  # safely broadcastable
        value = self.get_value(another_input)
        if len(shape) < len(perm) and value is None:
            raise RuntimeError(
                f"Unsupported node {node.name}: input {another_input} "
                "can not be broadcasted before transpose"
            )
        input_ind = list(node.input).index(another_input)
        if value is not None:
            # fold the constant
            while value.ndim < len(perm):
                value = value[None]
            value = value.transpose(inv_perm)
            cst_node = make_constant(f"{node.name}/{another_input}/const", value)
            node.input[input_ind] = cst_node.output[0]
            self += cst_node
        else:
            trans_node = make_node(
                "Transpose",
                [another_input],
                [f"{another_input}/transpose_output0"],
                name=f"{node.name}/Transpose",
                perm=inv_perm,
            )
            node.input[input_ind] = trans_node.output[0]
            self += trans_node

    def _rewrite_reaxis_op(self, graph: OnnxGraph, node: NodeProto, perm: List[int]):
        axis = self.get_attribute(node, "axis")
        if axis is not None:
            assert isinstance(axis, int)
            if axis < 0:
                axis += len(perm)
            new_axis = perm.index(axis)
            self.set_attribute(node, "axis", new_axis)
        elif node.op_type in ("Pad", "Slice"):
            axes: Optional[Sequence[int] | np.ndarray] = None
            if len(node.input) > 3:
                axes = self.get_value(node.input[3])
            else:
                node.input.append("")
            if axes is None:
                axes = list(range(len(perm)))
            axes = [i + len(perm) if i < 0 else i for i in axes]
            new_axes = [perm.index(i) for i in axes]
            axes_node = make_constant(f"{node.name}/axes", np.array(new_axes, np.int64))
            node.input[3] = axes_node.output[0]
            self += axes_node
            logger.debug(f"node {node.name} new pad axes: {new_axes}")
        elif node.op_type == "Tile":
            repeats = self.get_value(node.input[1])
            if repeats is None:
                raise RuntimeError(
                    f"Tile {node.name} repeats is dynamic, which is not supported"
                )
            new_repeats = [repeats[i] for i in perm]
            cst_node = make_constant(
                f"{node.name}/repeats", np.array(new_repeats, np.int64)
            )
            node.input[1] = cst_node.output[0]
            self += cst_node
            logger.debug(f"node {node.name} new repeats: {new_repeats}")
