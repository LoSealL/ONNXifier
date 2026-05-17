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

import onnx
from onnx.defs import OpSchema
from onnxscript import script
from onnxscript.onnx_opset import opset23 as op

from ...shape_inference import register_shape_inference
from .. import TRT_IR_DOMAIN
from .attention_plugin import _get_int_attribute

_T = "T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"]

# Define ONNX OpSchema for TRT DequantizeLinear
dequantize_linear_schema = OpSchema(
    # assumes every operator has a unique name 'DequantizeLinear' even across multiple
    # domains 'trt' and ''.
    name="trt::DequantizeLinear",
    domain=TRT_IR_DOMAIN.domain,
    since_version=TRT_IR_DOMAIN.version,
    doc=(
        "TensorRT DequantizeLinear operator for pseudo-quantized "
        "floating-point tensors. Unlike the standard ONNX DequantizeLinear, "
        "the input x is a pseudo-quantized floating-point tensor with the "
        "same data type as the output y."
    ),
    inputs=[
        OpSchema.FormalParameter(
            name="x",
            description="Pseudo-quantized input tensor (float).",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="x_scale",
            description="Scale tensor.",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="x_zero_point",
            description="Zero-point tensor (optional).",
            type_str=_T[0],
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="y",
            description="Dequantized output tensor.",
            type_str=_T[0],
        ),
    ],
    type_constraints=[
        (*_T, "Input and output data type."),
    ],
    attributes=[
        OpSchema.Attribute(
            name="axis",
            type=OpSchema.AttrType.INT,
            description="The axis of the quantization dimension. Default is 1.",
            required=False,
        ),
        OpSchema.Attribute(
            name="block_size",
            type=OpSchema.AttrType.INT,
            description=(
                "Block size for block-wise quantization. Default is 0 (per-tensor)."
            ),
            required=False,
        ),
    ],
)

onnx.defs.register_schema(dequantize_linear_schema)


def from_onnx_dequantize_linear(op: onnx.NodeProto) -> onnx.NodeProto:
    """
    Convert a standard ONNX DequantizeLinear (opset 21) node to TRT DequantizeLinear.

    ONNX DequantizeLinear inputs (opset 21):
        0: x (required) - Quantized tensor
        1: x_scale (required) - Scale tensor
        2: x_zero_point (optional) - Zero-point tensor

    ONNX DequantizeLinear outputs (opset 21):
        0: y (required) - Dequantized tensor

    ONNX DequantizeLinear attributes (opset 21):
        - axis: default 1
        - block_size: default 0

    TRT DequantizeLinear inputs:
        0: x (required) - Pseudo-quantized float tensor
        1: x_scale (required) - Scale tensor
        2: x_zero_point (optional) - Zero-point tensor

    TRT DequantizeLinear outputs:
        0: y (required) - Dequantized float tensor

    TRT DequantizeLinear attributes:
        - axis: default 1
        - block_size: default 0

    Args:
        op: The ONNX DequantizeLinear node to convert.

    Returns:
        A TRT domain DequantizeLinear node.
    """
    domain = TRT_IR_DOMAIN.domain
    plugin_op = onnx.NodeProto()
    plugin_op.op_type = "DequantizeLinear"
    plugin_op.domain = domain
    plugin_op.name = op.name

    # Map inputs directly: x, x_scale, x_zero_point
    plugin_op.input.append(op.input[0])
    plugin_op.input.append(op.input[1])
    if len(op.input) > 2:
        plugin_op.input.append(op.input[2])

    # Map output directly: y
    plugin_op.output.append(op.output[0])

    # Extract attributes with ONNX defaults
    axis = _get_int_attribute(op, "axis", 1)
    block_size = _get_int_attribute(op, "block_size", 0)

    axis_attr = onnx.AttributeProto()
    axis_attr.name = "axis"
    axis_attr.type = onnx.AttributeProto.INT
    axis_attr.i = axis
    plugin_op.attribute.append(axis_attr)

    block_size_attr = onnx.AttributeProto()
    block_size_attr.name = "block_size"
    block_size_attr.type = onnx.AttributeProto.INT
    block_size_attr.i = block_size
    plugin_op.attribute.append(block_size_attr)

    return plugin_op


@register_shape_inference(
    domain=dequantize_linear_schema.domain,
    # WA: remove "trt::" prefix
    op_type="DequantizeLinear",
)
@script(default_opset=op)
def trt_dequantize_linear_shape_infer(
    x,
    scales,
    zero_point,
    axis: int = 1,
    block_size: int = 0,
):
    """Shape inference for trt::DequantizeLinear (modelopt)."""
    return x
