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

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph, logger
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register("conv_dequantize_weight")
class ConvDequantizeWeightRewriter(Rewriter):
    """Fold int8 weight constant to DequantizeLinear op."""

    def __init__(self):
        # const->cast->mul->conv
        cast = SingleNodePattern("Cast")
        mul = SingleNodePattern("Mul")
        conv = SingleNodePattern("Conv")
        pattern = GraphPattern()
        pattern.add_edge(cast, mul)
        pattern.add_edge(mul, conv)
        super().__init__(pattern=pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        cast, mul, conv = nodes
        weight = self.get_value(cast.input[0])
        if weight is None:
            raise ValueError(f"Can't evaluate weight value: {cast.input[0]}")
        try:
            np.iinfo(weight.dtype)
        except ValueError:
            return  # weight is not quantized type
        # get another mul value
        scale_value = self.get_value_or_die(
            set(mul.input).difference({cast.output[0]}).pop()
        )
        scale_value = scale_value.squeeze()
        if scale_value.ndim == 0:
            scale_value = scale_value[None]
        zero_point = make_constant(
            f"{conv.name}/zero_point", np.zeros([scale_value.shape[0]], weight.dtype)
        )
        scale = make_constant(f"{conv.name}/scale", scale_value)
        dq = make_node(
            "DequantizeLinear",
            inputs=[cast.input[0], scale.output[0], zero_point.output[0]],
            outputs=[conv.input[1]],
            name=f"{conv.name}/DequantizeWeight",
            axis=0,
        )
        self += [dq, scale, zero_point]
        self -= [cast, mul]


@PASSES.register("gemm_dequantize_weight")
class GemmDequantizeWeightRewriter(Rewriter):
    """Fold int8 weight constant to DequantizeLinear op."""

    def __init__(self):
        # const->cast->mul->conv
        cast = SingleNodePattern("Cast")
        mul = SingleNodePattern("Mul")
        gemm = SingleNodePattern("MatMul") | SingleNodePattern("Gemm")
        transpose = SingleNodePattern("Transpose")
        pattern1 = GraphPattern().add_edge(cast, mul).add_edge(mul, gemm)
        pattern2 = (
            GraphPattern()
            .add_edge(cast, mul)
            .add_edge(mul, transpose)
            .add_edge(transpose, gemm)
        )
        super().__init__(pattern=pattern1 | pattern2)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        if len(nodes) == 3:
            cast, mul, gemm = nodes
            transpose = None
        else:
            cast, mul, transpose, gemm = nodes
        weight = self.get_value(cast.input[0])
        if weight is None:
            raise ValueError(f"Can't evaluate weight value: {cast.input[0]}")
        try:
            np.iinfo(weight.dtype)
        except ValueError:
            return  # weight is not quantized type
        # get another mul value
        scale_value = self.get_value_or_die(
            set(mul.input).difference({cast.output[0]}).pop()
        )
        scale_value = scale_value.squeeze()
        if scale_value.ndim == 0:
            scale_value = scale_value[None]
        zero_point = make_constant(
            f"{gemm.name}/zero_point", np.zeros([scale_value.shape[0]], weight.dtype)
        )
        scale = make_constant(f"{gemm.name}/scale", scale_value)
        if transpose is None:
            gemm_weight = mul.output[0]
        else:
            gemm_weight = transpose.input[0]
        dq = make_node(
            "DequantizeLinear",
            inputs=[cast.input[0], scale.output[0], zero_point.output[0]],
            outputs=[gemm_weight],
            name=f"{gemm.name}/DequantizeWeight",
            axis=0,
        )
        self += [dq, scale, zero_point]
        self -= [cast, mul]


@PASSES.register("recalculate_dequantize_weight")
class RecalculateDequantizeWeightRewriter(Rewriter):
    r"""Recalculate quantized weight for quantized conv.

    Before:

        DequantizeLinear -> Conv
                            /
               Const(F32)---

    After:

             DequantizeLinear -> Conv
                                   /
        Const(I8)->DequantizeLinear
    """

    def __init__(self):
        dq = SingleNodePattern("DequantizeLinear")
        conv = SingleNodePattern("Conv")
        pattern = GraphPattern()
        pattern.add_edge(dq, conv)
        super().__init__(pattern=pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        _, conv = nodes
        if op := self.get_input_node(conv, 1):
            if op.op_type == "DequantizeLinear":
                return  # skip if weight is already dequantized
        else:
            return

        weight = self.get_value(conv.input[1])
        if weight is None:
            raise ValueError(f"Can't evaluate weight value: {conv.input[1]}")
        qweight, scale_value, zp_value = self._symm_minmax_quantize(weight)
        if (err := self._quant_err(weight, qweight, scale_value, zp_value)) > 1:
            logger.info("skip requantize weight due to quant error exceeding 1")
            logger.debug(f"max err: {err}")
            return  # skip if quantization error is too large

        zero_point = make_constant(f"{conv.name}/zero_point", zp_value)
        scale = make_constant(f"{conv.name}/scale", scale_value)
        new_weight = make_constant(f"{conv.name}/new_weight", qweight)
        dq = make_node(
            "DequantizeLinear",
            inputs=[new_weight.output[0], scale.output[0], zero_point.output[0]],
            outputs=[f"{conv.name}/DequantizeWeight_output0"],
            name=f"{conv.name}/DequantizeWeight",
            axis=0,
        )
        conv.input[1] = dq.output[0]
        self += [dq, scale, zero_point, new_weight]

    def _symm_minmax_quantize(self, weight: np.ndarray):
        shape = weight.shape
        weight = weight.reshape([shape[0], -1])
        min_w = weight.min(1)
        max_w = weight.max(1)
        scale = np.maximum(-min_w, max_w) / 127.5
        zero_point = np.zeros_like(scale, np.int8)
        qweight = np.round(weight / scale[..., None]).astype(np.int32)
        qweight += zero_point[..., None]
        qweight = np.clip(qweight, -128, 127).astype(np.int8)
        qweight = qweight.reshape(shape)
        qweight[scale == 0, None] = 0
        scale[scale == 0] = 1
        return qweight, scale.astype(weight.dtype), zero_point

    def _quant_err(self, weight, qweight, scale, zero_point):
        scale = scale.reshape([-1, 1, 1, 1])
        zero_point = zero_point.reshape([-1, 1, 1, 1])
        reweight = weight / scale + zero_point
        err = np.abs(reweight - qweight).max()
        return err
