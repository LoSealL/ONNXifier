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

# pylint: disable=arguments-differ
from typing import List

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import Pattern, SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("activation_to_relu", deps=["fuse_swish", "fuse_mish"])
class ActivationToReluRewriter(Rewriter):
    """Rewrite any activation operator to relu operator."""

    _ACTIVATION_OPS = ["Sigmoid", "Swish", "Gelu", "Mish"]

    def __init__(self):
        pattern = sum([SingleNodePattern(op) for op in self._ACTIVATION_OPS])
        assert isinstance(pattern, Pattern)
        super().__init__(pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], target: str = "Relu"):
        assert target in ("Relu", "LeakyRelu", "PRelu")
        node = nodes[0]

        relu = make_node(target, node.input, node.output, name=node.name)

        self -= node
        self += relu
