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

from typing import List

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from .... import OnnxGraph
from ...pattern import SingleNodePattern
from ...rewriter import Rewriter
from . import OP_CONVERTER


@OP_CONVERTER.register("Mish")
class Mish(Rewriter):
    """Convert Mish according to expression:

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Mish"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        softplus = make_node(
            "Softplus",
            [node.input[0]],
            [f"{node.name}/Softplus_output0"],
            name=f"{node.name}/Softplus",
        )
        tanh_softplus = make_node(
            "Tanh",
            [softplus.output[0]],
            [f"{node.name}/Tanh_output0"],
            name=f"{node.name}/Tanh",
        )
        mul = make_node(
            "Mul",
            [node.input[0], tanh_softplus.output[0]],
            [node.output[0]],
            name=f"{node.name}/Mul",
        )
        self += [softplus, tanh_softplus, mul]
        self -= node
