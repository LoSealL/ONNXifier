"""
Copyright (C) 2026 The ONNXIFIER Authors.

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


@L2.register(name="split_to_slice", deps=["initializer_to_constant"])
class SplitToSliceRewriter(Rewriter):
    """Change Split node to Slice node."""

    def __init__(self):
        super().__init__(SingleNodePattern("Split"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        axis = self.get_attribute(node, "axis") or 0
        assert isinstance(axis, int)
        if len(node.input) < 2:
            num_outputs = self.get_attribute(node, "num_outputs")
            if num_outputs is None:
                num_outputs = graph.tensor_shape(node.input[0])[axis]
            assert isinstance(num_outputs, int)
            shape = graph.tensor_shape(node.input[0])[axis]
            assert isinstance(shape, int)
            if shape % num_outputs == 0:
                split = [shape // num_outputs] * num_outputs
            else:
                remind = shape % num_outputs
                split = [(shape - remind) // (num_outputs - 1)] * (num_outputs - 1)
                split.append(remind)
        else:
            split_node = self.get_input_node_or_die(node, 1)
            split = self.get_value_or_die(split_node).tolist()
        starts = 0
        for i, (ch, _) in enumerate(zip(split, node.output)):
            starts_node = make_constant(
                name=f"{node.name}/Starts{i}", value=np.array([starts], "int64")
            )
            ends_node = make_constant(
                name=f"{node.name}/Ends{i}", value=np.array([starts + ch], "int64")
            )
            axes_node = make_constant(
                name=f"{node.name}/Axes{i}", value=np.array([axis], "int64")
            )
            slice_node = make_node(
                op_type="Slice",
                inputs=[
                    node.input[0],
                    starts_node.output[0],
                    ends_node.output[0],
                    axes_node.output[0],
                ],
                outputs=[node.output[i]],
                name=f"{node.name}/Slice{i}",
            )
            self += [starts_node, ends_node, axes_node, slice_node]
            starts += ch
        self -= node
