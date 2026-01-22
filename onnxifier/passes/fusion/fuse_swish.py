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

import json
from typing import List

from onnx.helper import make_function, make_node, make_operatorsetid
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("fuse_swish")
class FuseSwishRewriter(Rewriter):
    r"""Fuse sigmoid-mul into Swish activation function.

     Swish is also known as SiLU (Sigmoid-Weighted Linear Unit) and is defined as:

    .. math::

         Swish(x) = x \cdot \sigma(x)

     where :math:`\sigma` is the sigmoid function.

     Before:

         a->sigmoid->mul->b
          \__________/

     After:

         a->swish(function)->b
    """

    def __init__(self):
        pattern = GraphPattern()
        node_a = SingleNodePattern()
        sigm = SingleNodePattern("Sigmoid")
        mul = SingleNodePattern("Mul")
        pattern.add_edge(node_a, sigm)
        pattern.add_edge(node_a, mul)
        pattern.add_edge(sigm, mul)
        super().__init__(pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        _, sigmoid, mul = nodes
        swish = make_node(
            "Swish",
            inputs=[sigmoid.input[0]],
            outputs=[mul.output[0]],
            name=f"{sigmoid.name}/Swish",
            domain="onnxifier",  # domain can not be empty or "ai.onnx"
            doc_string=json.dumps([sigmoid.name, mul.name]),
        )
        self += [swish]
        self -= [sigmoid, mul]

        swish_func = make_function(
            "onnxifier",
            "Swish",
            inputs=[sigmoid.input[0]],
            outputs=[mul.output[0]],
            nodes=[sigmoid, mul],
            opset_imports=[
                make_operatorsetid("", graph.opset_version),
                make_operatorsetid("onnxifier", 1),
            ],
        )
        graph.onnx_add_function(swish_func)
