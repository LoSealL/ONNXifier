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

# pylint: disable=arguments-differ

from typing import List

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import L2
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@L2.register("splittosequence_to_slice")
class SplitToSequenceToSliceRewriter(Rewriter):
    """Convert SplitToSequence and SequenceAt to slice operator."""

    def __init__(self):
        super().__init__(SingleNodePattern("SplitToSequence"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        sequence_consumers = graph.onnx_successors(node)
        axis = self.get_attribute(node, "axis", 0)
        keepdims = self.get_attribute(node, "keepdims", 1)
        assert isinstance(axis, int) and isinstance(keepdims, int)
        axis = int(axis)
        keepdims = bool(keepdims)
        input_shape = graph.tensor_shape(node.input[0])
        axis_dim = input_shape[axis]
        assert isinstance(axis_dim, int)

        if len(node.input) > 1:
            split_arr = self.get_value_or_die(node.input[1])
            if split_arr.ndim == 0:
                split = [int(split_arr)] * (axis_dim // int(split_arr))
            else:
                split = [int(i) for i in split_arr]
            assert sum(split) == input_shape[axis]
        else:
            split = [1] * axis_dim

        for i in sequence_consumers:
            if i.op_type != "SequenceAt":
                raise NotImplementedError(
                    f"{i.op_type} after SplitToSequence is not supported now"
                )

            self.sequence_at_to_slice(graph, i, split, axis, keepdims)
        # replace SplitToSequence with Identity
        self += make_node("Identity", [node.input[0]], [node.output[0]], name=node.name)
        self -= node

    def sequence_at_to_slice(
        self,
        graph: OnnxGraph,
        node: NodeProto,
        split: List[int],
        axis: int,
        keepdims: bool,
    ):
        """Convert SequenceAt to slice operator."""

        position = self.get_value_or_die(node.input[1])
        starts = sum(split[:position])
        ends = starts + split[position]

        starts_node = make_constant(
            name=f"{node.name}/Starts{position}", value=np.array([starts], "int64")
        )
        ends_node = make_constant(
            name=f"{node.name}/Ends{position}", value=np.array([ends], "int64")
        )
        axes_node = make_constant(
            name=f"{node.name}/Axes{position}", value=np.array([axis], "int64")
        )
        slice_node = make_node(
            "Slice",
            [
                node.input[0],
                starts_node.output[0],
                ends_node.output[0],
                axes_node.output[0],
            ],
            node.output,
            name=node.name,
        )
        if not keepdims and split[position] == 1:
            # add a squeeze
            squeeze_node = make_node(
                "Squeeze",
                [f"{node.name}/Squeeze_input0"],
                [node.output[0]],
                name=f"{node.name}/Squeeze",
            )
            slice_node.output[0] = squeeze_node.input[0]
            self += squeeze_node
        self += [starts_node, ends_node, axes_node, slice_node]
