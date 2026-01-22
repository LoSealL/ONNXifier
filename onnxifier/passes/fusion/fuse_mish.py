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

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("fuse_mish")
class FuseMishRewriter(Rewriter):
    r"""Fuse Mish activation function.

     Mish is defined as:

    .. math::

         mish(x) = x \cdot tanh(softplus(x))

     where :math:`softplus(x) = ln(1 + exp(x))`.

     Before:

         a->softplus->tanh->mul->b
          \_________________/

     After:

         a->mish(function)->b
    """

    def __init__(self):
        pattern = GraphPattern()
        node_a = SingleNodePattern()
        softp = SingleNodePattern("Softplus")
        tanh = SingleNodePattern("Tanh")
        mul = SingleNodePattern("Mul")
        pattern.add_edge(node_a, softp)
        pattern.add_edge(softp, tanh)
        pattern.add_edge(tanh, mul)
        pattern.add_edge(node_a, mul)
        super().__init__(pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        _, softplus, tanh, mul = nodes
        mish = make_node(
            "Mish",
            inputs=[softplus.input[0]],
            outputs=[mul.output[0]],
            name=f"{softplus.name}/Mish",
        )
        self += [mish]
        self -= [softplus, tanh, mul]
