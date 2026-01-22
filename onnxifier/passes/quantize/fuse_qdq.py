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

from typing import Collection, List

import numpy as np
import onnx
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("fuse_qconv", deps=["infer_shape"])
class FuseQConvRewriter(Rewriter):
    """Fuse Quantize-Dequantize to QLinearConv.

    Before:

        DequantizeLinear -----
                              | (weight)
        DequantizeLinear -> Conv -> QuantizeLinear
                             /
        DequantizeLinear ___/  (bias, optional)

    After:

        QLinearConv

    Note:

        Bias typestr in QLinearConv must only be int32,
    """

    def __init__(self):
        qconv = GraphPattern()
        conv = SingleNodePattern("Conv")
        qconv.add_edge(SingleNodePattern("DequantizeLinear"), conv)
        qconv.add_edge(SingleNodePattern("DequantizeLinear"), conv)
        qconv.add_edge(conv, SingleNodePattern("QuantizeLinear"))
        super().__init__(qconv)

    def _unpack_nodes(self, nodes: Collection[NodeProto]):
        nodes = set(nodes)
        conv = next(filter(lambda n: n.op_type == "Conv", nodes))
        qinput = self.get_input_node(conv, 0)
        qweight = self.get_input_node(conv, 1)
        qoutput = None
        for n in self.get_output_node(conv, 0):
            if n.op_type == "QuantizeLinear":
                qoutput = n
        assert qinput in nodes and qinput is not None
        assert qweight in nodes and qweight is not None
        assert qoutput in nodes and qoutput is not None
        return qinput, qweight, conv, qoutput

    def _conv_attrs(self, conv):
        keys = ("auto_pad", "dilations", "group", "kernel_shape", "pads", "strides")
        attrs = {k: self.get_attribute(conv, k) for k in keys}
        return {k: v for k, v in attrs.items() if v is not None}

    def _canonical_bias(self, graph: OnnxGraph, conv, x_scale, w_scale):
        if (bias_value := self.get_value(conv.input[2])) is not None:
            # bias is a constant float
            # quantize bias to int32 implicitly
            scale = self.get_value_or_die(x_scale) * self.get_value_or_die(w_scale)
            qbias_value = np.round(bias_value / scale).astype(np.int32)
            graph.initializer.append(
                onnx.numpy_helper.from_array(qbias_value, conv.name + "/qbias")
            )
            return conv.name + "/qbias"
        if (bias := self.get_input_node_or_die(conv, 2)).op_type == "DequantizeLinear":
            # bias is already been quantized
            bias_value = self.get_value(bias.input[0])
            if bias_value is None:
                raise RuntimeError(
                    f"bias value from DequantizeLinear({bias.name}) is None"
                )
            if bias_value.dtype == np.int32:
                return bias.input[0]
            raise TypeError(
                f"QLinearConv only supports int32 bias, got {bias_value.dtype}"
            )
        if graph.tensor_type(conv.input[2]) != onnx.TensorProto.INT32:
            raise TypeError("QLinearConv only supports int32 bias.")
        return conv.input[2]

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        qinput, qweight, conv, qoutput = self._unpack_nodes(nodes)
        if len(graph.onnx_successors(qinput)) > 1:
            # this means input dequantize fans out to multiple downstream nodes,
            # we can't fuse it to QLinearConv
            return
        x = qinput.input[0]
        x_scale = qinput.input[1]
        x_zero_point = qinput.input[2]
        w = qweight.input[0]
        w_scale = qweight.input[1]
        w_zero_point = qweight.input[2]
        y_scale = qoutput.input[1]
        y_zero_point = qoutput.input[2]
        inputs = [
            x,
            x_scale,
            x_zero_point,
            w,
            w_scale,
            w_zero_point,
            y_scale,
            y_zero_point,
        ]
        if len(conv.input) == 3:
            bias = self._canonical_bias(graph, conv, x_scale, w_scale)
            inputs.append(bias)
        qconv = make_node(
            "QLinearConv",
            inputs=inputs,
            outputs=[qoutput.output[0]],
            name=conv.name + "/qconv",
            **self._conv_attrs(conv),  # type: ignore
        )
        self += qconv
        self -= [qinput, qweight, conv, qoutput]


@PASSES.register("unfuse_qconv")
class UnfuseQConvRewriter(Rewriter):
    """Unfuse QLinearConv to Quantize-Dequantize.

    Before:

        QLinearConv

    After:

        DequantizeLinear -----
                              | (weight)
        DequantizeLinear -> Conv -> QuantizeLinear
                             /
        DequantizeLinear ___/  (bias, optional)
    """

    def __init__(self):
        super().__init__(SingleNodePattern("QLinearConv"))

    def _conv_attrs(self, conv):
        keys = ("auto_pad", "dilations", "group", "kernel_shape", "pads", "strides")
        attrs = {k: self.get_attribute(conv, k) for k in keys}
        return {k: v for k, v in attrs.items() if v is not None}

    def _canonical_bias(self, graph: OnnxGraph, qconv):
        x_scale_value = self.get_value(qconv.input[1])
        w_scale_value = self.get_value(qconv.input[4])
        if x_scale_value is None:
            raise RuntimeError(f"x_scale value from {qconv.name} is not constant")
        if w_scale_value is None:
            raise RuntimeError(f"w_scale value from {qconv.name} is not constant")
        bias_value = self.get_value_or_die(qconv.input[8])
        bias_value = bias_value.astype(np.float32) * (x_scale_value * w_scale_value)
        graph.initializer.append(
            onnx.numpy_helper.from_array(bias_value, qconv.name + "/dq_bias")
        )
        return qconv.name + "/dq_bias"

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        x = node.input[0]
        x_scale = node.input[1]
        x_zero_point = node.input[2]
        w = node.input[3]
        w_scale = node.input[4]
        w_zero_point = node.input[5]
        y_scale = node.input[6]
        y_zero_point = node.input[7]
        if len(node.input) == 9:
            bias = [self._canonical_bias(graph, node)]
        else:
            bias = []
        dqinput = make_node(
            "DequantizeLinear",
            inputs=[x, x_scale, x_zero_point],
            outputs=[x + "_dq"],
            name=node.name + "/dqinput",
            axis=1,
        )
        dqweight = make_node(
            "DequantizeLinear",
            inputs=[w, w_scale, w_zero_point],
            outputs=[w + "_dq"],
            name=node.name + "/dqweight",
            axis=0,
        )
        conv = make_node(
            "Conv",
            inputs=[x + "_dq", w + "_dq"] + bias,
            outputs=[node.output[0] + "_q"],
            name=node.name + "/conv",
            **self._conv_attrs(node),  # type: ignore
        )
        qoutput = make_node(
            "QuantizeLinear",
            inputs=[node.output[0] + "_q", y_scale, y_zero_point],
            outputs=[node.output[0]],
            name=node.name + "/qoutput",
            axis=1,
        )
        self += [conv, dqinput, dqweight, qoutput]
        self -= node


@PASSES.register("unfuse_qmatmul")
class UnfuseQMatMulRewriter(Rewriter):
    """Unfuse QLinearMatMul to Quantize-Dequantize.

    Before:

        QLinearMatMul

    After:

        DequantizeLinear -----
                              | (weight)
        DequantizeLinear -> MatMul -> QuantizeLinear
                             /
        DequantizeLinear ___/  (bias, optional)
    """

    def __init__(self):
        super().__init__(SingleNodePattern("QLinearMatMul"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        a = node.input[0]
        a_scale = node.input[1]
        a_zero_point = node.input[2]
        b = node.input[3]
        b_scale = node.input[4]
        b_zero_point = node.input[5]
        y_scale = node.input[6]
        y_zero_point = node.input[7]
        dq_a = make_node(
            "DequantizeLinear",
            inputs=[a, a_scale, a_zero_point],
            outputs=[a + "_dq"],
            name=node.name + "/dqa",
        )
        dq_b = make_node(
            "DequantizeLinear",
            inputs=[b, b_scale, b_zero_point],
            outputs=[b + "_dq"],
            name=node.name + "/dqb",
        )
        matmul = make_node(
            "MatMul",
            inputs=[a + "_dq", b + "_dq"],
            outputs=[node.output[0] + "_q"],
            name=node.name + "/matmul",
        )
        qoutput = make_node(
            "QuantizeLinear",
            inputs=[node.output[0] + "_q", y_scale, y_zero_point],
            outputs=[node.output[0]],
            name=node.name + "/qoutput",
        )
        self += [dq_a, dq_b, matmul, qoutput]
        self -= node
