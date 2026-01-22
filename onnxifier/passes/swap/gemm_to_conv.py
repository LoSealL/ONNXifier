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

import math
from typing import List

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(name="gemm_to_conv", deps=["initializer_to_constant"])
class GEMMToConvRewrite(Rewriter):
    """Convert GEMM op to Conv"""

    def __init__(self):
        super().__init__(SingleNodePattern("Gemm") | SingleNodePattern("MatMul"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        gemm_node = nodes[0]
        self._convert_fc_to_conv(graph, gemm_node)
        self._convert_NDim_matmul_to_conv(graph, gemm_node)

    def _convert_fc_to_conv(self, graph: OnnxGraph, gemm_node: NodeProto):
        node = gemm_node.name

        data_shape = graph.static_tensor_shape(gemm_node.input[0])
        weight_shape = graph.static_tensor_shape(gemm_node.input[1])
        alpha = self.get_attribute(gemm_node, "alpha") or 1.0
        beta = self.get_attribute(gemm_node, "beta") or 1.0
        transA = self.get_attribute(gemm_node, "transA") or 0
        transB = self.get_attribute(gemm_node, "transB") or 0
        if len(data_shape) != 2 or len(weight_shape) != 2 or transA:
            return

        # data reshape
        data_shape_cst = make_constant(
            name=f"{node}/DataShape", value=np.array(data_shape + [1, 1], dtype="int64")
        )
        data_reshape_node = make_node(
            op_type="Reshape",
            inputs=[
                gemm_node.input[0],
                data_shape_cst.output[0],
            ],
            outputs=[f"{node}/Conv_input0"],
            name=f"{node}/DataReshape",
        )

        # weight reshape fold
        weight_node = self.get_input_node_or_die(gemm_node, 1)
        if weight_node.op_type == "Constant":
            # fold const
            weight_value = self.get_value_or_die(weight_node).copy()
            weight_value *= alpha
            if not transB:
                weight_value = weight_value.T
            new_weight_node = make_constant(
                name=f"{node}/Weight",
                value=weight_value[..., None, None],
            )
            self -= weight_node
            weight_port = new_weight_node.output[0]
        elif weight_node.op_type == "DequantizeLinear":
            quant_weight_node = self.get_input_node_or_die(weight_node, 0)
            quant_weight = self.get_value_or_die(quant_weight_node)
            if not transB:
                quant_weight = quant_weight.T
                # must quantize on output channels
                assert self.get_attribute(weight_node, "axis") == 1
                self.set_attribute(weight_node, "axis", 0)
            new_weight_node = make_constant(
                f"{node}/QWeight", quant_weight[..., None, None]
            )
            weight_node.input[0] = new_weight_node.output[0]
            weight_port = weight_node.output[0]
            self -= quant_weight_node
            if alpha != 1:
                # rescale with alpha
                scale_node = self.get_input_node_or_die(weight_node, 1)
                scale_value = self.get_value_or_die(scale_node).copy()
                scale_value *= alpha
                scale_const_node = make_constant(f"{node}/scale", scale_value)
                self -= scale_node
                self += scale_const_node
                weight_node.input[1] = scale_const_node.output[0]
        else:
            return

        conv_inputs = [data_reshape_node.output[0], weight_port]

        # bias
        if len(gemm_node.input) == 3:
            bias_node = self.get_input_node_or_die(gemm_node, 2)
            if bias_node.op_type == "Constant":
                bias_value = self.get_value_or_die(bias_node) * beta
                if bias_value.ndim > 1:
                    bias_value = bias_value.squeeze()
                bias_const_node = make_constant(f"{node}/bias", bias_value)
                self -= bias_node
                self += bias_const_node
                conv_inputs.append(bias_const_node.output[0])
            else:
                conv_inputs.append(gemm_node.input[2])

        # conv
        conv_node = make_node(
            op_type="Conv",
            inputs=conv_inputs,
            outputs=[f"{node}/Conv_out"],
            name=f"{node}/Conv",
        )
        self.set_attribute(conv_node, "dilations", [1, 1])
        self.set_attribute(conv_node, "group", 1)
        self.set_attribute(conv_node, "kernel_shape", [1, 1])
        self.set_attribute(conv_node, "pads", [0, 0, 0, 0])
        self.set_attribute(conv_node, "strides", [1, 1])

        # out reshape
        out_shape = graph.static_tensor_shape(gemm_node.output[0])
        out_shape_cst = make_constant(
            name=f"{node}/OutShape", value=np.array(out_shape, dtype="int64")
        )
        out_reshape_node = make_node(
            op_type="Reshape",
            inputs=[
                conv_node.output[0],
                out_shape_cst.output[0],
            ],
            outputs=gemm_node.output[:],
            name=f"{node}/OutReshape",
        )

        self += [
            data_shape_cst,
            data_reshape_node,
            new_weight_node,
            conv_node,
            out_shape_cst,
            out_reshape_node,
        ]
        self -= gemm_node

    def _split_factor(self, number):
        """Split number into two factors, such that the factor1 * factor2 == number"""
        factor = 1
        for i in range(1, int(math.sqrt(number)) + 1):
            if number % i == 0:
                factor = i
        return factor, number // factor

    def _convert_NDim_matmul_to_conv(self, graph: OnnxGraph, gemm_node: NodeProto):
        """It just convert
            [1, B, M, K]x[1, 1, K, N]
        to
            [1, K, B_M', B_M''] @ [N, K, 1, 1]
        It will convert data and weight of matmul to inp and filter of conv
        """
        node = gemm_node.name

        data_shape = graph.static_tensor_shape(gemm_node.input[0])
        weight_shape = graph.static_tensor_shape(gemm_node.input[1])

        # check
        # support unequal number of dimensions
        if len(data_shape) <= 2 and len(weight_shape) <= 2:
            return
        # if both data and weight are batched, it should be split
        data_batch = np.prod(data_shape[:-1])  # type: ignore
        weight_batch = np.prod(weight_shape[:-2])  # type: ignore
        if data_batch != 1 and weight_batch != 1:
            return

        # swap AxB to B'xA'
        swap_operands = data_batch == 1
        # if A is [B, M, K], treat it as [B * M, 1, K]
        B, M, N, K = (
            data_batch * weight_batch,
            1,
            weight_shape[-1],
            data_shape[-1],
        )

        # prepare data input of conv
        if swap_operands:
            # treat weight as input of conv
            inp_input = gemm_node.input[1]
            # [1, 1, M, K]x[1, B, K, N] => [1, K, B, N]@[M, K, 1, 1]
            # [1, B, K, N] => [1, K, B, N]
            n = len(weight_shape)
            perm = [n - 2] + list(range(n - 2)) + [n - 1]
            f1, f2 = self._split_factor(B * N)
        else:
            # treat data as input of conv
            inp_input = gemm_node.input[0]
            # [1, B, M, K]x[1, 1, K, N] => [1, K, B, M]@[N, K, 1, 1]
            # [1, B, M, K] => [1, K, B, M]
            n = len(data_shape)
            perm = [n - 1] + list(range(n - 2)) + [n - 2]
            f1, f2 = self._split_factor(B * M)
        inp_shape = [1, K, f1, f2]
        inp_permute = make_node(
            "Transpose",
            inputs=[inp_input],
            outputs=[f"{node}_data_permute"],
            name=f"{node}/DataPermute",
            perm=perm,
        )
        inp_reshape_cst = make_constant(
            name=f"{node}/DataReshapeCst", value=np.array(inp_shape, dtype="int64")
        )
        inp_reshape = make_node(
            "Reshape",
            inputs=[inp_permute.output[0], inp_reshape_cst.output[0]],
            outputs=[f"{node}_DataReshape"],
            name=f"{node}/DataReshape",
        )
        self += [inp_permute, inp_reshape_cst, inp_reshape]

        # prepare filter of conv
        if swap_operands:
            # treat data as filter of conv
            filter_input = gemm_node.input[0]
            # [1, 1, M, K]x[1, B, K, N] => [1, K, B, N]@[M, K, 1, 1]
            # [1, 1, M, K] => [M, K, 1, 1]
            filter_reshape_cst = make_constant(
                name=f"{node}/filterShape", value=np.array([M, K, 1, 1], dtype="int64")
            )
            filter_node = make_node(
                "Reshape",
                inputs=[filter_input, filter_reshape_cst.output[0]],
                outputs=[f"{node}_filter_reshape"],
                name=f"{node}/FilterReshape",
            )
            self += filter_reshape_cst
        else:
            # treat weight as filter of conv
            filter_input = gemm_node.input[1]
            # [1, B, M, K]x[1, 1, K, N] => [1, K, B, M]@[N, K, 1, 1]
            # [1, 1, K, N] => [N, K, 1, 1]
            perm = [3, 2, 0, 1]
            if len(weight_shape) != 4:
                filter_reshape_cst = make_constant(
                    name=f"{node}/filterShape",
                    value=np.array([1, 1, K, N], dtype="int64"),
                )
                filter_reshape = make_node(
                    "Reshape",
                    inputs=[filter_input, filter_reshape_cst.output[0]],
                    outputs=[f"{node}_filter_reshape"],
                    name=f"{node}/FilterReshape",
                )
                filter_input = filter_reshape.output[0]
                self += [filter_reshape_cst, filter_reshape]
            filter_node = make_node(
                "Transpose",
                inputs=[filter_input],
                outputs=[f"{node}_filter_permute"],
                name=f"{node}/FilterPermute",
                perm=perm,
            )
        self += filter_node

        conv_inputs = [inp_reshape.output[0], filter_node.output[0]]

        # conv
        conv_node = make_node(
            op_type="Conv",
            inputs=conv_inputs,
            outputs=[f"{node}/Conv_out"],
            name=f"{node}/Conv",
            dilations=[1, 1],
            group=1,
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        )

        # out reshape and permute
        out_shape = graph.static_tensor_shape(gemm_node.output[0])  # [1, B, M, N]
        if swap_operands:
            # output of conv is [1, M, B, N]
            perm = [0, 2, 1, 3]
            temp_out_shape = [1, M, B, N]
        else:
            # output of conv is [1, N, B, M]
            perm = [0, 2, 3, 1]
            temp_out_shape = [1, N, B // data_shape[-2], data_shape[-2]]
        temp_out_shape_cst = make_constant(
            name=f"{node}/TempOutShape", value=np.array(temp_out_shape, dtype="int64")
        )
        temp_out_reshape = make_node(
            op_type="Reshape",
            inputs=[
                conv_node.output[0],
                temp_out_shape_cst.output[0],
            ],
            outputs=[f"{node}/temp_out_reshape"],
            name=f"{node}/TempOutReshape",
        )
        self += [
            conv_node,
            temp_out_shape_cst,
            temp_out_reshape,
        ]
        if len(out_shape) == 4:
            out_permute = make_node(
                "Transpose",
                inputs=[temp_out_reshape.output[0]],
                outputs=gemm_node.output[:],
                name=f"{node}/out_permute",
                perm=perm,
            )
            self += out_permute
        else:
            # reshape to NDim array
            out_permute = make_node(
                "Transpose",
                inputs=[temp_out_reshape.output[0]],
                outputs=[f"{node}_out_permute"],
                name=f"{node}/out_permute",
                perm=perm,
            )
            out_shape_cst = make_constant(
                name=f"{node}/OutShape", value=np.array(out_shape, dtype="int64")
            )
            out_reshape_node = make_node(
                op_type="Reshape",
                inputs=[
                    out_permute.output[0],
                    out_shape_cst.output[0],
                ],
                outputs=gemm_node.output[:],
                name=f"{node}/OutReshape",
            )
            self += [
                out_permute,
                out_shape_cst,
                out_reshape_node,
            ]
        self -= gemm_node
