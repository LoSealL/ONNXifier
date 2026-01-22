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

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import L1
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@L1.register(name="prelu_to_leaky", deps=["initializer_to_constant"])
class PReluToLeakyRewriter(Rewriter):
    """Convert PRelu whose parameter is a scaler to LeakyRelu."""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("PRelu"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        # quick check slope shape
        slope_shape = graph.tensor_shape(node.input[1])
        if len(slope_shape) >= 1 and slope_shape[0] != 1:
            # keep PRelu
            return

        slope_value = None
        for pred in graph.onnx_predecessors(node):
            if node.input[1] in pred.output:
                slope_value = self.get_value(pred)
                break
        if slope_value is None:
            # slope is not static
            return

        leakyrelu_node = make_node(
            "LeakyRelu",
            inputs=node.input[:1],
            outputs=node.output,
            name=node.name + "/to_leaky",
            alpha=float(slope_value.squeeze()),
        )
        self += leakyrelu_node
        self -= node
