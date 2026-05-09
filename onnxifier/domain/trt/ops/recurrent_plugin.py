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

from ....domain.shape_inference import register_shape_inference
from .. import TRT_IR_DOMAIN

_T = "T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"]
_U = "U", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"]
_T_CL = "T_CL", ["tensor(int32)", "tensor(int64)"]


gated_delta_rule_schema = OpSchema(
    name="GatedDeltaRule",
    domain=TRT_IR_DOMAIN.domain,
    since_version=TRT_IR_DOMAIN.version,
    doc="Qwen3.5 Gated Delta Net plugin with explicit recurrent state output.",
    inputs=[
        OpSchema.FormalParameter(
            name="q",
            description="Query [n, seq, h, k]",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="k",
            description="Key [n, seq, h, k]",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="v",
            description="Value [n, seq, hv, v]",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="gate",
            description="A gating tensor [n, seq, hv]",
            type_str=_U[0],  # gate may differ from qkv type
        ),
        OpSchema.FormalParameter(
            name="beta",
            description="B gating tensor [n, seq, hv]",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="past_ssm_state",
            description="Recurrent state in [n, hv, k, v]",
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
            description="GDN output",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="present_ssm_state",
            description="Updated recurrent state [n, hv, k, v]",
            type_str=_T[0],
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
    ],
    type_constraints=[
        (*_T, "Input QKV data type."),
        (*_U, "Gate data type."),
        (*_T_CL, "Context lengths data type."),
    ],
    attributes=[
        OpSchema.Attribute(
            name="k_dim",
            type=OpSchema.AttrType.INT,
            description="Dimension of key head",
            required=True,
        ),
        OpSchema.Attribute(
            name="v_dim",
            type=OpSchema.AttrType.INT,
            description="Dimension of value head",
            required=True,
        ),
        OpSchema.Attribute(
            name="num_v_heads",
            type=OpSchema.AttrType.INT,
            description="Number value heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="use_qk_l2norm",
            default_value=make_attribute(
                "use_qk_l2norm", 1, "Use QK L2 normalization", onnx.AttributeProto.INT
            ),
            description="Use QK L2 normalization",
        ),
    ],
)

onnx.defs.register_schema(gated_delta_rule_schema)


@register_shape_inference(
    domain=gated_delta_rule_schema.domain,
    op_type=gated_delta_rule_schema.name,
)
@onnxscript.script(default_opset=op)
def gated_delta_rule_shape_infer(
    q,
    k,
    v,
    gate,
    beta,
    past_ssm_state,
    context_lengths,
    k_dim: int,
    v_dim: int,
    num_v_heads: int,
    use_qk_l2norm: int = 1,
):
    """Shape inference for trt::GatedDeltaRule.

    output[0] follows v input shape (input[2]).
    output[1] passes through past_ssm_state unchanged.
    """

    q_shape = op.Shape(q)  # type: ignore
    v_shape = op.Shape(v)  # type: ignore
    out_shape = op.Concat(q_shape[:-1], v_shape[-1:], axis=0)  # type: ignore
    out = op.CastLike(op.ConstantOfShape(out_shape), q)  # type: ignore
    return out, op.Identity(past_ssm_state)  # type: ignore
