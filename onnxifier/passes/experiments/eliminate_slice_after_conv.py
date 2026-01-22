"""
Copyright (C) 2024 The ONNXIFIER Authors.

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

from copy import deepcopy
from itertools import chain
from typing import List

import networkx as nx
from onnx import NodeProto

from ... import OnnxGraph
from ...logger import debug
from .. import PASSES
from ..pattern import SingleNodePattern, StartEndPointPattern
from ..rewriter import Rewriter
from ..utils import is_elewise, make_constant


@PASSES.register("eliminate_slice_after_conv", deps=["split_to_slice"])
class EliminateSliceAfterConvRewriter(Rewriter):
    r"""Eliminate slice(split) after conv by splitting output channels of conv
    into multiple pieces.

    Before:

        conv -> * -> split -> A
                          \__ B

    After:

        conv_1 -> * -> A
        conv_2 -> * -> B

    Note:
        The split must be taken on channel axis.
    """

    def __init__(self):
        # TODO: add ConvTranspose support
        super().__init__(pattern=SingleNodePattern("Conv"))

    def is_slice_on_channel(self, node: NodeProto, graph: OnnxGraph) -> bool:
        """Whether the slice is on channel axis only (axis=1).

        output = Slice(input, starts, ends, axes, steps)
        """

        assert node.op_type == "Slice"
        if len(node.input) >= 4:
            axes = self.get_value(node.input[3])
            if axes is None:
                # dynamic axes
                return False
            if len(axes) == 1:
                # whether on channel axis
                return axes[0] == 1
        else:
            axes = None
        data_shape, _ = graph.tensor_info(node.input[0])
        if data_shape is None:
            # dynamic data shape
            return False
        if axes is None:
            axes = list(range(len(data_shape)))
        if len(axes) > 1 and 1 in axes:
            starts = self.get_value(node.input[1])
            ends = self.get_value(node.input[2])
            if starts is None or ends is None:
                # dynamic slice
                return False
            data_shape, _ = graph.tensor_info(node.input[0])
            if data_shape is None:
                # dynamic data shape
                return False
            steps = (
                self.get_value(node.input[4])
                if len(node.input) >= 5
                else [1 for _ in axes]
            )
            if steps is None:
                # dynamic steps
                return False
            for start, end, axis, step in zip(starts, ends, axes, steps):
                if axis == 1:
                    continue
                if start != 0 or end != data_shape[axis] or step != 1:
                    # not on channel axis
                    return False
            return True
        return False

    def get_sliced_axes(self, node: NodeProto, graph: OnnxGraph) -> slice:
        """Get the axes that are sliced by the slice node."""
        assert node.op_type == "Slice"
        data_shape = graph.tensor_shape(node.input[0])
        axes = self.get_value(node.input[3]) if len(node.input) >= 4 else None
        if axes is None:
            axes = list(range(len(data_shape)))
        steps = (
            self.get_value(node.input[4]) if len(node.input) >= 5 else [1 for _ in axes]
        )
        starts = self.get_value(node.input[1])
        ends = self.get_value(node.input[2])
        assert starts is not None and ends is not None and steps is not None

        for start, end, axis, step in zip(starts, ends, axes, steps):
            if axis != 1:
                continue
            return slice(start, end, step)

        raise RuntimeError("Slice is not on channel axis.")

    def make_split_conv(self, conv: NodeProto, axes: slice, graph: OnnxGraph):
        """Split the output channels of conv into multiple pieces."""
        # TODO: support quantized conv
        weights = self.get_value(conv.input[1])
        if weights is None:
            raise RuntimeError("Quantized conv or dynamic weight is not supported.")
        bias = self.get_value(conv.input[2]) if len(conv.input) >= 3 else None
        new_weights = weights[axes]
        new_bias = bias[axes] if bias is not None else None
        node_conv = deepcopy(conv)
        node_conv.output[0] = f"{conv.name}/{axes.start}_{axes.stop}_{axes.step}_output"
        node_conv.name = f"{conv.name}/{axes.start}_{axes.stop}_{axes.step}"
        node_weight = make_constant(f"{node_conv.name}/weight", new_weights)
        node_conv.input[1] = node_weight.output[0]
        self += node_weight
        if new_bias is not None:
            node_bias = make_constant(f"{node_conv.name}/bias", new_bias)
            node_conv.input[2] = node_bias.output[0]
            self += node_bias
        self += node_conv
        return node_conv

    def copy_relay_nodes(
        self, conv: NodeProto, nodes: List[NodeProto], graph: OnnxGraph
    ):
        """Duplicate the relay nodes between conv (start) and slice (end) node."""
        # TODO: support sibling nodes.
        # E.g. Add(a, b) where a is split and b is channel-wise constant.
        h = graph.onnx_subgraph(nodes)
        start, end, *relay_nodes = nodes
        new_relay_nodes = {i.name: i for i in deepcopy(relay_nodes)}
        for node in new_relay_nodes.values():
            node.name += f"/{conv.name}"
            for i, _ in enumerate(node.input):
                node.input[i] += f"/{conv.name}"
            for i, _ in enumerate(node.output):
                node.output[i] += f"/{conv.name}"
        for node in h.onnx_successors(start):
            new_node = new_relay_nodes[node.name]
            for i, input_name in enumerate(node.input):
                if input_name == start.output[0]:
                    new_node.input[i] = conv.output[0]
        for node in h.onnx_predecessors(end):
            new_node = new_relay_nodes[node.name]
            for i, output_name in enumerate(node.output):
                if output_name == end.input[0]:
                    new_node.output[i] = end.output[0]
        self += list(new_relay_nodes.values())

    def rewrite(self, graph, nodes, *args, **kwargs):
        conv = nodes[0]
        search_results = self.search_slices(conv, graph)
        for _, post_nodes in enumerate(search_results):
            slice_nodes = []
            relay_nodes = []
            post_nodes: List[NodeProto] = list(post_nodes)
            for relay_node in post_nodes:
                if relay_node.name == conv.name:
                    continue
                if not is_elewise(relay_node):
                    if relay_node.op_type == "Slice":
                        slice_nodes.append(relay_node)
                    else:
                        slice_nodes.clear()  # invalid branch found, skip
                        break
                else:
                    relay_nodes.append(relay_node)
            if len(slice_nodes) == 1:
                if not self.is_slice_on_channel(slice_nodes[0], graph):
                    continue
            else:
                continue
            axes = self.get_sliced_axes(slice_nodes[0], graph)
            new_conv = self.make_split_conv(conv, axes, graph)
            debug(f"relay nodes: {[i.name for i in relay_nodes]}")
            self.copy_relay_nodes(new_conv, [conv, slice_nodes[0], *relay_nodes], graph)
            self -= slice_nodes

    def search_slices(self, conv: NodeProto, graph: OnnxGraph):
        """Search for slice nodes after conv."""
        h = graph.onnx_subgraph(chain(nx.descendants(graph, conv.name), [conv]))
        # limit search depth to 10
        subgraph_nodes = [conv.name]
        for child in h:
            if nx.shortest_path_length(h, conv.name, child) <= 10:
                subgraph_nodes.append(child)
        # renew h
        h = graph.onnx_subgraph(subgraph_nodes)
        search_results = StartEndPointPattern(
            start_pattern=SingleNodePattern(conv.op_type, conv.name),
            end_pattern=SingleNodePattern("Slice"),
        ).match(h)
        yield from search_results
