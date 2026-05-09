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
from onnxscript.onnx_opset import opset23 as op
from onnxscript.values import Opset

from ...shape_inference import register_shape_inference
from .. import TRT_IR_DOMAIN

_T = "T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"]
_T_KV = "T_KV", ["tensor(float16)", "tensor(bfloat16)", "tensor(float8e4m3fn)"]

# Define ONNX OpSchema for AttentionPlugin
attention_plugin_schema = OpSchema(
    name="AttentionPlugin",
    domain=TRT_IR_DOMAIN.domain,
    since_version=TRT_IR_DOMAIN.version,
    doc="TensorRT attention plugin with RoPE, KV cache, and attention computation.",
    inputs=[
        OpSchema.FormalParameter(
            name="q",
            description="Query tensor",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="k",
            description="Key tensor",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="v",
            description="Value tensor",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="past_key_value",
            description="KV cache tensor",
            type_str=_T_KV[0],
        ),
        OpSchema.FormalParameter(
            name="context_lengths",
            description="Context length tensor",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="rope_rotary_cos_sin",
            description="RoPE rotary embeddings (FP32)",
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="kvcache_start_index",
            description=(
                "KV cache start index tensor of shape [kv_cache_start_batch_size]"
            ),
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="attention_mask",
            description="Attention mask tensor (optional)",
            type_str="tensor(int32)",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
        OpSchema.FormalParameter(
            name="k_v_scale_quant_orig",
            description=(
                "Packed KV dequant scales for FP8 KV cache. "
                "Shape [2] float: [k_scale_quant_orig, v_scale_quant_orig] (optional)"
            ),
            type_str="tensor(float)",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output tensor",
            type_str=_T[0],
        ),
        OpSchema.FormalParameter(
            name="present_key_value",
            description=(
                "Updated KV cache tensor with dynamic shape "
                "[batch_size, 2, num_kv_heads, present_kv_cache_len, head_size]"
            ),
            type_str=_T_KV[0],
        ),
    ],
    type_constraints=[
        (*_T, "Input Q/K/V data type."),
        (*_T_KV, "KV cache data type."),
    ],
    attributes=[
        OpSchema.Attribute(
            name="num_q_heads",
            type=OpSchema.AttrType.INT,
            description="Number of query heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="num_kv_heads",
            type=OpSchema.AttrType.INT,
            description="Number of key-value heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="head_size",
            type=OpSchema.AttrType.INT,
            description="Size of each attention head",
            required=True,
        ),
        OpSchema.Attribute(
            name="enable_tree_attention",
            type=OpSchema.AttrType.INT,
            description="Whether to enable tree attention (0(false), 1(true))",
            required=True,
        ),
        OpSchema.Attribute(
            name="enable_fp8_kv_cache",
            type=OpSchema.AttrType.INT,
            description="Whether to use FP8 KV cache (0(false), 1(true)). Optional.",
            required=False,
        ),
        OpSchema.Attribute(
            name="sliding_window_size",
            type=OpSchema.AttrType.INT,
            description=(
                "Sliding window size for attention (-1 = no sliding window, "
                ">0 = window size)."
            ),
            required=False,
        ),
    ],
)

onnx.defs.register_schema(attention_plugin_schema)


def _trim(inputs):
    """Helper to trim trailing empty inputs."""
    trimmed = list(inputs)
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    return trimmed


def from_onnx_attention(
    op: onnx.NodeProto,
    head_size: int,
    *,
    enable_tree_attention: int = 0,
    enable_fp8_kv_cache: int = 0,
    sliding_window_size: int = -1,
    context_lengths: str = "",
    rope_rotary_cos_sin: str = "",
    kvcache_start_index: str = "",
    k_v_scale_quant_orig: str = "",
) -> onnx.NodeProto:
    """
    Convert a standard ONNX Attention (opset 24) node to AttentionPlugin.

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

    AttentionPlugin inputs:
        q, k, v, past_key_value, context_lengths, rope_rotary_cos_sin,
        kvcache_start_index, attention_mask (opt), attention_pos_id (opt),
        k_v_scale_quant_orig (opt)

    AttentionPlugin outputs:
        attn_output, present_key_value

    Args:
        op: The ONNX Attention node to convert.
        head_size: Required attention head size provided by caller.
        enable_tree_attention: Whether to enable tree attention (0=false, 1=true).
            Does not exist in standard ONNX Attention - use as kwarg.
        enable_fp8_kv_cache: Whether to use FP8 KV cache (0=false, 1=true).
            Does not exist in standard ONNX Attention - use as kwarg.
        sliding_window_size: Sliding window size (-1=no sliding window, >0=window size).
            Does not exist in standard ONNX Attention - use as kwarg.
        context_lengths: Input name for context length tensor.
            Does not exist in standard ONNX Attention - use as kwarg.
        rope_rotary_cos_sin: Input name for RoPE rotary embeddings.
            Does not exist in standard ONNX Attention - use as kwarg.
        kvcache_start_index: Input name for KV cache start index tensor.
            Does not exist in standard ONNX Attention - use as kwarg.
        k_v_scale_quant_orig: Input name for FP8 KV cache dequant scales.
            Does not exist in standard ONNX Attention - use as kwarg.
    """
    domain = TRT_IR_DOMAIN.domain
    plugin_op = onnx.NodeProto()
    plugin_op.op_type = "AttentionPlugin"
    plugin_op.domain = domain
    plugin_op.name = op.name

    # Map Q, K, V inputs (ONNX inputs 0, 1, 2)
    op_inputs = [op.input[0], op.input[1], op.input[2]]

    # past_key_value: use past_key (input[4]) if present
    past_key = op.input[4] if len(op.input) > 4 else ""
    if past_key:
        assert len(op.input) > 5  # past_value also needed
        op_inputs.append(past_key)
    else:
        op_inputs.append("")

    # context_lengths, rope_rotary_cos_sin, kvcache_start_index - from kwargs
    op_inputs.append(context_lengths)
    op_inputs.append(rope_rotary_cos_sin)
    op_inputs.append(kvcache_start_index)

    # attention_mask - from ONNX input[3] (attn_mask)
    attn_mask = op.input[3] if len(op.input) > 3 else ""
    op_inputs.append(attn_mask)

    # attention_pos_id, k_v_scale_quant_orig - from kwargs
    op_inputs.append(k_v_scale_quant_orig)
    plugin_op.input.extend(_trim(op_inputs))

    # Extract attributes from ONNX Attention (spec: q_num_heads, kv_num_heads)
    num_q_heads = _get_int_attribute(op, "q_num_heads", 0)
    num_kv_heads = _get_int_attribute(op, "kv_num_heads", num_q_heads)
    if head_size <= 0:
        raise ValueError(
            "head_size must be a positive integer for AttentionPlugin conversion."
        )

    # Set required attributes on plugin op
    num_heads_attr = onnx.AttributeProto()
    num_heads_attr.name = "num_q_heads"
    num_heads_attr.type = onnx.AttributeProto.INT
    num_heads_attr.i = num_q_heads
    plugin_op.attribute.append(num_heads_attr)

    num_kv_heads_attr = onnx.AttributeProto()
    num_kv_heads_attr.name = "num_kv_heads"
    num_kv_heads_attr.type = onnx.AttributeProto.INT
    num_kv_heads_attr.i = num_kv_heads
    plugin_op.attribute.append(num_kv_heads_attr)

    head_size_attr = onnx.AttributeProto()
    head_size_attr.name = "head_size"
    head_size_attr.type = onnx.AttributeProto.INT
    head_size_attr.i = int(head_size)
    plugin_op.attribute.append(head_size_attr)

    enable_tree_attr = onnx.AttributeProto()
    enable_tree_attr.name = "enable_tree_attention"
    enable_tree_attr.type = onnx.AttributeProto.INT
    enable_tree_attr.i = enable_tree_attention
    plugin_op.attribute.append(enable_tree_attr)

    enable_fp8_attr = onnx.AttributeProto()
    enable_fp8_attr.name = "enable_fp8_kv_cache"
    enable_fp8_attr.type = onnx.AttributeProto.INT
    enable_fp8_attr.i = enable_fp8_kv_cache
    plugin_op.attribute.append(enable_fp8_attr)

    sliding_window_attr = onnx.AttributeProto()
    sliding_window_attr.name = "sliding_window_size"
    sliding_window_attr.type = onnx.AttributeProto.INT
    sliding_window_attr.i = sliding_window_size
    plugin_op.attribute.append(sliding_window_attr)

    # Map outputs:
    # ONNX output[0] (Y) -> attn_output
    # ONNX output[1] (present_key) -> present_key_value
    plugin_op.output.append(op.output[0])
    plugin_op.output.append(op.output[1] if len(op.output) > 1 else "")

    return plugin_op


def _get_int_attribute(node: onnx.NodeProto, name: str, default: int) -> int:
    """Helper to get an integer attribute from a node."""
    for attr in node.attribute:
        if attr.name == name:
            return attr.i
    return default


@register_shape_inference(
    domain=attention_plugin_schema.domain,
    op_type=attention_plugin_schema.name,
)
@onnxscript.script(Opset(attention_plugin_schema.domain, 1), default_opset=op)
def attention_plugin_shape_inference(
    q,
    k,
    v,
    past_key_value,
    context_lengths,
    rope_rotary_cos_sin,
    kvcache_start_index,
    attention_mask,
    k_v_scale_quant_orig,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    enable_tree_attention: int = 0,
    enable_fp8_kv_cache: int = 0,
    sliding_window_size: int = -1,
):
    """Shape inference for trt::AttentionPlugin.

    output[0] (attn_output) follows q input shape.
    output[1] (present_key_value) passes through past_key_value unchanged.
    """

    outs = op.Attention(
        q,
        k,
        v,
        kv_num_heads=num_kv_heads,
        q_num_heads=num_q_heads,
    )
    out = outs[0]  # type: ignore
    return out, op.Identity(past_key_value)
