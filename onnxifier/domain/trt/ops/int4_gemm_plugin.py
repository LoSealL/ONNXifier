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

from onnx.defs import OpSchema, register_schema
from onnxscript import script
from onnxscript.onnx_opset import opset23 as op

from ...shape_inference import register_shape_inference
from .. import TRT_IR_DOMAIN

_T = "T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"]

int4_gemm_plugin_schema = OpSchema(
    name="Int4GroupwiseGemmPlugin",
    domain=TRT_IR_DOMAIN.domain,
    since_version=TRT_IR_DOMAIN.version,
    doc="Custom TensorRT Int4 Groupwise GEMM plugin for GPTQ quantized models.",
    inputs=[
        OpSchema.FormalParameter(
            name="input",
            description="Input tensor",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="qweight",
            description="Quantized weight tensor, int4 packed into int8",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="scales",
            description="Scale tensor",
            type_str=_T[0],
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="Output tensor",
            type_str=_T[0],
        ),
    ],
    type_constraints=[
        (*_T, "Input and output data type."),
    ],
    attributes=[
        OpSchema.Attribute(
            name="gemm_n",
            type=OpSchema.AttrType.INT,
            description="Output feature dimension",
            required=True,
        ),
        OpSchema.Attribute(
            name="gemm_k",
            type=OpSchema.AttrType.INT,
            description="Input feature dimension",
            required=True,
        ),
        OpSchema.Attribute(
            name="group_size",
            type=OpSchema.AttrType.INT,
            description="Group size for groupwise quantization",
            required=True,
        ),
    ],
)

register_schema(int4_gemm_plugin_schema)


@register_shape_inference(
    domain=int4_gemm_plugin_schema.domain,
    op_type=int4_gemm_plugin_schema.name,
)
@script(default_opset=op)
def int4_gemm_plugin_shape_infer(
    inputs,
    qweight,
    scales,
    gemm_n: int = 0,
    gemm_k: int = 0,
    group_size: int = 0,
):
    """Shape inference for trt::Int4GroupwiseGemmPlugin.

    qweight was packed into 8-bit. out = inputs @ weight.
    """
    k = op.Constant(value_int=gemm_k)
    n = op.Constant(value_int=gemm_n)
    w_shape = op.Concat(
        op.Unsqueeze(k, 0),  # type: ignore
        op.Unsqueeze(n, 0),  # type: ignore
        axis=0,
    )
    fake_w = op.CastLike(
        op.ConstantOfShape(w_shape),  # type: ignore
        inputs,
    )
    return op.MatMul(inputs, fake_w)  # type: ignore
