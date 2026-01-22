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

from typing import List

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(name="merge_mul_to_conv", deps=["initializer_to_constant"])
class MergeMulToConvRewrite(Rewriter):
    """Merge multiply to the nearest convolution

    Pattern1:

        mul
         |
        add/sub
         |
        conv

    Pattern2:

        conv
         |
        mul

    After:

        conv(bias)
    """

    def __init__(self):
        pattern1 = GraphPattern()
        pattern2 = GraphPattern()
        mul = SingleNodePattern("Mul")
        elt = SingleNodePattern()
        conv = SingleNodePattern("Conv")

        pattern1.add_edge(mul, elt)
        pattern1.add_edge(elt, conv)
        pattern2.add_edge(conv, mul)
        super().__init__(pattern1 | pattern2)

    def _get_input_and_factor(self, mul: NodeProto, graph: OnnxGraph):
        inp0, inp1 = mul.input
        for node in graph:
            pb_node = graph.nodes[node]["pb"]
            # skip not cst op
            if pb_node.op_type != "Constant":
                continue
            # input 0 is cst
            if pb_node.output[0] == inp0:
                return inp1, self.get_value(pb_node)
            # input 1 is cst
            elif pb_node.output[0] == inp1:
                return inp0, self.get_value(pb_node)
        return None, None

    def _make_div(self, input_name: str, op_name: str, factor: np.ndarray):
        cst = make_constant(op_name + "/DivCst", factor)
        div = make_node(
            op_type="Div",
            inputs=[input_name, cst.output[0]],
            outputs=[op_name + "/DivOutput"],
            name=op_name + "/Div",
        )
        return cst, div

    def _make_mul(self, input_name: str, op_name: str, factor: np.ndarray):
        cst = make_constant(op_name + "/MulCst", factor)
        mul = make_node(
            op_type="Mul",
            inputs=[input_name, cst.output[0]],
            outputs=[op_name + "/MulOutput"],
            name=op_name + "/Mul",
        )
        return cst, mul

    def rewrite_pattern1(self, graph: OnnxGraph, nodes: List[NodeProto]):
        """Fuse mul-elt-conv to conv"""
        mul, elt, conv = nodes

        if elt.op_type not in {"Add", "Sub"}:
            return

        # which is constant of mul
        input_name, factor = self._get_input_and_factor(mul, graph)
        if input_name is None or factor is None:
            return

        # which is the input of add/sub op
        in_idx = 0 if mul.output[0] == elt.input[0] else 1

        # change add/sub op input and make new div op
        elt.input[in_idx] = input_name
        cst1, div1 = self._make_div(elt.input[1 - in_idx], elt.name, factor)
        elt.input[1 - in_idx] = div1.output[0]
        self += [cst1, div1]

        # change conv's weight and bias
        cst2, mul2 = self._make_mul(conv.input[1], conv.input[1], factor)
        conv.input[1] = mul2.output[0]
        self += [cst2, mul2]

        self -= [mul]

    def rewrite_pattern2(self, graph: OnnxGraph, nodes: List[NodeProto]):
        """Fuse conv-mul to conv"""
        conv, mul = nodes
        input_name, factor = self._get_input_and_factor(mul, graph)
        if input_name is None or factor is None:
            return
        if len(graph.onnx_successors(conv)) > 1:
            # multi fan-out
            return

        # can not broadcast to weight
        if factor.ndim > 1 and np.prod(factor.shape[-2:]) > 1:
            return

        # change weight
        factor = factor.reshape([-1, 1, 1, 1])
        cst, weight_mul = self._make_mul(conv.input[1], conv.input[1], factor)
        conv.input[1] = weight_mul.output[0]
        # change bias
        if len(conv.input) > 2 and conv.input[2]:
            bias_shape = graph.static_tensor_shape(conv.input[2])
            bias_shape[0] = -1
            factor = factor.reshape(bias_shape)
            cst2, bias_mul = self._make_mul(conv.input[2], conv.input[2], factor)
            self += [cst2, bias_mul]
            conv.input[2] = bias_mul.output[0]
        # for post_mul in graph.onnx_successors(mul):
        #     post_mul.input[list(post_mul.input).index(mul.output[0])] = conv.output[0]
        conv.output[0] = mul.output[0]
        self += [cst, weight_mul, conv]
        self -= mul

    # pylint: disable=arguments-differ
    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        if len(nodes) == 3:
            self.rewrite_pattern1(graph, nodes)
        elif len(nodes) == 2:
            self.rewrite_pattern2(graph, nodes)
