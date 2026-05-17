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

import ml_dtypes
import numpy as np
import onnx
from onnx import NodeProto

from .... import OnnxGraph, logger
from ....domain.trt import TRT_IR_DOMAIN
from ....domain.trt.ops.dequantize_linear import from_onnx_dequantize_linear
from ... import PASSES
from ...pattern import SingleNodePattern
from ...rewriter import Rewriter
from ...utils import make_constant
from . import EnsureTensorRTDomain

_FLOAT_TYPES = {
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.FLOAT16,
    onnx.TensorProto.BFLOAT16,
}


def _get_np_dtype_and_range(onnx_dtype: int):
    """Map ONNX quantized dtype to numpy dtype and valid [min, max] range."""
    dt = onnx.helper.tensor_dtype_to_np_dtype(onnx_dtype)
    try:
        dinfo = ml_dtypes.finfo(dt)
    except ValueError:
        dinfo = ml_dtypes.iinfo(dt)
    return dt, float(dinfo.min), float(dinfo.max)


def _expand_for_block_size(
    arr: np.ndarray,
    target_shape: tuple[int, ...],
    axis: int,
    block_size: int,
) -> np.ndarray:
    """Expand arr by repeating elements along axis to match target_shape."""
    if block_size <= 0:
        return _reshape_per_axis(arr, target_shape, axis)
    expanded = np.repeat(arr, block_size, axis=axis)
    slices = [slice(None)] * expanded.ndim
    slices[axis] = slice(0, target_shape[axis])
    return expanded[tuple(slices)]


def _reshape_per_axis(
    arr: np.ndarray,
    target_shape: tuple[int, ...],
    axis: int,
) -> np.ndarray:
    """Reshape 1-D per-axis scale/zp to broadcast-compatible shape.

    e.g. scale [K] with axis=0, target [K, N] → reshape to [K, 1].
    """
    if arr.ndim <= 1 and arr.size > 1:
        shape = [1] * len(target_shape)
        shape[axis] = arr.shape[0]
        return arr.reshape(shape)
    return arr


def _dequantize_to_float(
    x: np.ndarray,
    scale: np.ndarray,
    zero_point: np.ndarray | None,
    axis: int,
    block_size: int,
    out_dtype: int,
) -> np.ndarray:
    """Compute pseudo-quantized float values from quantized integer x."""
    target_shape = x.shape
    scale = _expand_for_block_size(scale, target_shape, axis, block_size)
    if zero_point is not None:
        zero_point = _expand_for_block_size(zero_point, target_shape, axis, block_size)

    x_float = x.astype(np.float32)
    if zero_point is not None:
        x_float = x_float - zero_point.astype(np.float32)
    result = x_float * scale.astype(np.float32)
    return result.astype(onnx.helper.tensor_dtype_to_np_dtype(out_dtype))


def _quantize_from_float(
    x: np.ndarray,
    scale: np.ndarray,
    zero_point: np.ndarray | None,
    axis: int,
    block_size: int,
    out_dtype: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute quantized values from pseudo-quantized float x."""
    target_shape = x.shape
    near_zero = np.abs(scale) < np.finfo(scale.dtype).tiny
    if near_zero.any():
        logger.warning(
            "%d scale values are zero or near-zero; "
            "those channels will be quantized to 0",
            int(near_zero.sum()),
        )
        scale = scale.copy()
        scale[near_zero] = 1.0

    np_dtype, qmin, qmax = _get_np_dtype_and_range(out_dtype)
    expand_scale = _expand_for_block_size(scale, target_shape, axis, block_size)
    if zero_point is not None:
        zero_point = _expand_for_block_size(zero_point, target_shape, axis, block_size)
    expand_mask = _expand_for_block_size(near_zero, target_shape, axis, block_size)
    x = np.where(expand_mask, 0.0, x)
    scale = np.where(near_zero, 0.0, scale)  # remask scale to zero
    x_inv = x.astype(np.float32) / expand_scale.astype(np.float32)
    if zero_point is not None:
        x_inv = x_inv + zero_point.astype(np.float32)
    return np.rint(x_inv).clip(qmin, qmax).astype(np_dtype), scale


def _infer_axis(
    x_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
    block_size: int,
) -> int:
    """Infer the quantization axis from scale and x shapes.

    For per-axis quantization, scale shape is either 1-D (e.g. [K])
    or N-D with axis dim matching x and other dims = 1 (e.g. [K, 1]).
    For blocked quantization, scale_shape[axis] == x_shape[axis] // block_size.
    """
    if block_size <= 0:
        if len(scale_shape) == 1 and scale_shape[0] > 1:
            for a, s in enumerate(x_shape):
                if s == scale_shape[0]:
                    return a
        if len(scale_shape) == len(x_shape):
            for a, s in enumerate(x_shape):
                if scale_shape[a] == s and scale_shape[a] > 1:
                    remaining = [i for i in range(len(x_shape)) if i != a]
                    if all(scale_shape[r] == 1 for r in remaining):
                        return a
        return 1
    for a, s in enumerate(x_shape):
        if scale_shape[a] * block_size == s:
            return a
    return 1


@PASSES.register(name="trt_dequantize_linear_replace")
class TRTDequantizeLinearRewriter(EnsureTensorRTDomain):
    """Replace ONNX DequantizeLinear node with TensorRT DequantizeLinear.

    Since TRT's DequantizeLinear expects input x to be a pseudo-quantized
    floating-point tensor, the original low-bit integer input is converted to
    float by numpy computation (x - zero_point) * scale and baked as a
    Constant node.
    """

    def __init__(self):
        pattern = SingleNodePattern("DequantizeLinear")
        super().__init__(pattern=pattern)

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto]):
        node = nodes[0]
        x_name = node.input[0]
        if not x_name:
            return

        x_shape, x_dtype = graph.tensor_info(x_name)
        if x_dtype in _FLOAT_TYPES:
            trt_x_name = x_name
        else:
            scale_name = node.input[1] if len(node.input) > 1 else ""
            zp_name = node.input[2] if len(node.input) > 2 else ""
            x_val = self.get_value(x_name)
            scale_val = self.get_value(scale_name) if scale_name else None
            zp_val = self.get_value(zp_name) if zp_name else None
            if x_val is None or scale_val is None:
                logger.debug("Skip %s: x or scale is not constant", node.name)
                return

            _, out_dtype = graph.tensor_info(node.output[0])
            if out_dtype == onnx.TensorProto.UNDEFINED and scale_name:
                _, out_dtype = graph.tensor_info(scale_name)
            if out_dtype == onnx.TensorProto.UNDEFINED:
                out_dtype = onnx.TensorProto.FLOAT

            axis = self.get_attribute(node, "axis", 1)
            block_size = self.get_attribute(node, "block_size", 0)
            pseudo_float = _dequantize_to_float(
                x_val, scale_val, zp_val, axis, block_size, out_dtype
            )
            const_node = make_constant(f"{node.name}/pseudo_quant", pseudo_float)
            self += const_node
            if x_shape is not None:
                graph.set_value_info(const_node.output[0], x_shape, out_dtype)
            trt_x_name = const_node.output[0]

        plugin_op = from_onnx_dequantize_linear(node)
        plugin_op.input[0] = trt_x_name
        self -= node
        self += plugin_op


@PASSES.register(name="trt_dequantize_linear_to_onnx")
class TRTDequantizeLinearToOnnxRewriter(Rewriter):
    """Restore TRT DequantizeLinear node back to ONNX DequantizeLinear.

    If the TRT input x is a constant floating-point tensor, it will be
    re-quantized to integer via numpy and baked as a Constant node.
    """

    def __init__(self):
        pattern = SingleNodePattern("DequantizeLinear").with_domain(
            TRT_IR_DOMAIN.domain
        )
        pattern |= SingleNodePattern("trt::DequantizeLinear").with_domain(
            TRT_IR_DOMAIN.domain
        )
        super().__init__(pattern=pattern)

    def rewrite(  # pylint: disable=arguments-differ
        self,
        graph: OnnxGraph,
        nodes: list[NodeProto],
        to_dtype: int = onnx.TensorProto.INT8,
    ):
        node = nodes[0]
        x_name = node.input[0]
        if not x_name:
            return

        x_shape, x_dtype = graph.tensor_info(x_name)
        scale_name = node.input[1] if len(node.input) > 1 else ""
        zp_name = node.input[2] if len(node.input) > 2 else ""

        onnx_x_name = x_name
        axis = None
        block_size = 0
        if x_dtype in _FLOAT_TYPES:
            x_val = self.get_value(x_name)
            scale_val = self.get_value(scale_name) if scale_name else None
            zp_val = self.get_value(zp_name) if zp_name else None
            if x_val is not None and scale_val is not None:
                block_size = self.get_attribute(node, "block_size", 0)
                axis = _infer_axis(x_val.shape, scale_val.shape, block_size)
                quant_val, scale_val = _quantize_from_float(
                    x_val, scale_val, zp_val, axis, block_size, to_dtype
                )
                quant_cst = make_constant(f"{node.name}/weight", quant_val)
                scale_cst = make_constant(f"{node.name}/scale", scale_val)
                self += [quant_cst, scale_cst]
                if x_shape is not None:
                    graph.set_value_info(quant_cst.output[0], x_shape, to_dtype)
                onnx_x_name = quant_cst.output[0]
                scale_name = scale_cst.output[0]

        onnx_op = onnx.NodeProto()
        onnx_op.op_type = "DequantizeLinear"
        onnx_op.domain = ""
        onnx_op.name = node.name
        onnx_op.input.append(onnx_x_name)
        if scale_name:
            onnx_op.input.append(scale_name)
        if zp_name:
            onnx_op.input.append(zp_name)
        onnx_op.output.extend(node.output)

        if axis is not None:
            axis_attr = onnx.AttributeProto()
            axis_attr.name = "axis"
            axis_attr.type = onnx.AttributeProto.INT
            axis_attr.i = axis
            onnx_op.attribute.append(axis_attr)
        if block_size > 0:
            bs_attr = onnx.AttributeProto()
            bs_attr.name = "block_size"
            bs_attr.type = onnx.AttributeProto.INT
            bs_attr.i = block_size
            onnx_op.attribute.append(bs_attr)

        self -= node
        self += onnx_op
