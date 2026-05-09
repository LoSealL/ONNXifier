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
import onnxscript
from onnx.defs import OpSchema
from onnx.helper import make_attribute
from onnxscript.onnx_opset import opset19 as op
from onnxscript.values import Opset

from ....domain.shape_inference import register_shape_inference
from .. import TRT_IR_DOMAIN

_T = "T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"]
_T_CL = "T_CL", ["tensor(int32)", "tensor(int64)"]


causal_conv1d_schema = OpSchema(
    name="CausalConv1d",
    domain=TRT_IR_DOMAIN.domain,
    since_version=TRT_IR_DOMAIN.version,
    doc="TensorRT causal 1D depthwise convolution plugin with persistent state.",
    inputs=[
        OpSchema.FormalParameter(
            name="x",
            description="Input tensor",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="weight",
            description="Conv weight",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="bias",
            description="Conv bias",
            type_str=_T[0],
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
        OpSchema.FormalParameter(
            name="past_conv_state",
            description="Conv state [batch, dim, kernel]",
            type_str=_T[0],
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
        OpSchema.FormalParameter(
            name="context_lengths",
            description="Per-batch actual token count [batch]",
            type_str=_T_CL[0],
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="Conv output",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="present_conv_state",
            description="Updated conv state",
            type_str=_T[0],
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
    ],
    type_constraints=[
        (*_T, "Input QKV data type."),
        (*_T_CL, "Context lengths data type."),
    ],
    attributes=[
        OpSchema.Attribute(
            name="padding",
            type=OpSchema.AttrType.INT,
            description="Padding",
            required=True,
        ),
        OpSchema.Attribute(
            name="groups",
            type=OpSchema.AttrType.INT,
            description="Groups",
            required=True,
        ),
        OpSchema.Attribute(
            name="dilation",
            description="Dilation",
            default_value=make_attribute("dilation", 1),
        ),
        OpSchema.Attribute(
            name="stride",
            description="Stride",
            default_value=make_attribute("stride", 1),
        ),
    ],
)

onnx.defs.register_schema(causal_conv1d_schema)


@register_shape_inference(op_type=causal_conv1d_schema.name)
@onnxscript.script(Opset(causal_conv1d_schema.domain, 1), default_opset=op)
def causal_conv1d_shape_inference(
    x,
    weight,
    bias,
    past_conv_state,
    context_lengths,
    padding: int,
    groups: int,
    dilation: int = 1,
    stride: int = 1,
):
    """Shape inference for trt::CausalConv1d.

    Returns x directly so ONNX shape inference copies x's shape to
    output[0].  output[1] passes through past_conv_state unchanged.
    """
    ## WA1: onnx can't support assignment like
    ## op.Constant(value_ints=[dilation, dilation])
    ## WA2: expression below can't pass shape inference, it will give
    ## [unk_1, unk_2, ...] because complex computation is not supported.
    ## We ignore pads and stride here. Assume it's padding same with
    ## stride=1 always.

    # x_shape = op.Shape(x)
    # w_shape = op.Shape(weight)
    # length = op.Unsqueeze(x_shape[-1] + padding - w_shape[-1] + 1, axes=0)
    # channels = op.Unsqueeze(w_shape[1] * groups, axes=0)
    # batch = op.Unsqueeze(x_shape[0], axes=0)
    # out_shape = op.Concat(batch, channels, length, axis=0)
    # out = op.CastLike(op.ConstantOfShape(out_shape), x)
    return x, op.Identity(past_conv_state)
