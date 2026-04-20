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

from .. import TRT_IR_DOMAIN
from .attention_plugin import _get_int_attribute, _trim

_T = "T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"]

# Define ONNX OpSchema for ViTAttentionPlugin
vit_attention_plugin_schema = OpSchema(
    name="ViTAttentionPlugin",
    domain=TRT_IR_DOMAIN.domain,
    since_version=TRT_IR_DOMAIN.version,
    doc="TensorRT ViT attention plugin (separate Q/K/V, no KV cache, no RoPE).",
    inputs=[
        OpSchema.FormalParameter(
            name="q",
            description="Query tensor in head-major layout [total_S, H, D]",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="k",
            description="Key tensor in head-major layout [total_S, H, D]",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="v",
            description="Value tensor in head-major layout [total_S, H, D]",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="cu_seqlens",
            description="Prefix sum of sequence lengths (int32, shape [B+1])",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="max_seqlen_carrier",
            description=(
                "Shape-only input used to carry runtime max sequence length hint; "
                "tensor values are ignored."
            ),
            type_str="tensor(int32)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output tensor [total_S, H, D]",
            type_str=_T[0],
        ),
    ],
    type_constraints=[
        (*_T, "Input Q/K/V data type."),
    ],
    attributes=[
        OpSchema.Attribute(
            name="num_heads",
            type=OpSchema.AttrType.INT,
            description="Number of attention heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="head_size",
            type=OpSchema.AttrType.INT,
            description="Size of each attention head",
            required=True,
        ),
    ],
)

onnx.defs.register_schema(vit_attention_plugin_schema)


def from_onnx_attention(
    op: onnx.NodeProto,
    head_size: int,
    *,
    cu_seqlens: str = "",
    max_seqlen_carrier: str = "",
) -> onnx.NodeProto:
    """
    Convert a standard ONNX Attention (opset 24) node to ViTAttentionPlugin.

    ViTAttentionPlugin is designed for Vision Transformer attention with
    head-major layout [total_S, H, D] for Q/K/V, no KV cache, and no RoPE.

    ONNX Attention inputs (opset 24):
        0: Q (required) - Query tensor
        1: K (optional) - Key tensor
        2: V (optional) - Value tensor
        3: attn_mask (optional) - Attention mask
        4: past_key (optional) - Past state cache for key
        5: past_value (optional) - Past state cache for value

    ONNX Attention outputs (opset 24):
        0: Y (required) - Output tensor
        1: present_key (optional) - Updated key cache
        2: present_value (optional) - Updated value cache

    ViTAttentionPlugin inputs:
        q, k, v, cu_seqlens, max_seqlen_carrier

    ViTAttentionPlugin outputs:
        attn_output

    Args:
        op: The ONNX Attention node to convert.
        head_size: Required attention head size provided by caller.
        cu_seqlens: Input name for cumulative sequence lengths (int32, shape [B+1]).
            Does not exist in standard ONNX Attention - use as kwarg.
        max_seqlen_carrier: Input name for max sequence length hint (shape-only).
            Does not exist in standard ONNX Attention - use as kwarg.
    """
    domain = TRT_IR_DOMAIN.domain
    plugin_op = onnx.NodeProto()
    plugin_op.op_type = "ViTAttentionPlugin"
    plugin_op.domain = domain
    plugin_op.name = op.name

    # Map Q, K, V inputs (ONNX inputs 0, 1, 2)
    op_inputs = [op.input[0], op.input[1], op.input[2]]

    # cu_seqlens, max_seqlen_carrier - from kwargs
    op_inputs.append(cu_seqlens)
    op_inputs.append(max_seqlen_carrier)
    plugin_op.input.extend(_trim(op_inputs))

    # Extract attributes from ONNX Attention (spec: q_num_heads)
    num_heads = _get_int_attribute(op, "q_num_heads", 0)
    if head_size <= 0:
        raise ValueError(
            "head_size must be a positive integer for ViTAttentionPlugin conversion."
        )

    # Set required attributes on plugin op
    num_heads_attr = onnx.AttributeProto()
    num_heads_attr.name = "num_heads"
    num_heads_attr.type = onnx.AttributeProto.INT
    num_heads_attr.i = num_heads
    plugin_op.attribute.append(num_heads_attr)

    head_size_attr = onnx.AttributeProto()
    head_size_attr.name = "head_size"
    head_size_attr.type = onnx.AttributeProto.INT
    head_size_attr.i = int(head_size)
    plugin_op.attribute.append(head_size_attr)

    # Map output: ONNX output[0] (Y) -> attn_output
    plugin_op.output.append(op.output[0])

    return plugin_op
