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

# pylint: disable=missing-function-docstring

import onnx
from onnx import AttributeProto

from onnxifier.domain.trt.ops.attention_plugin import from_onnx_attention


def _make_attention_node(
    inputs,
    outputs,
    num_heads=4,
    kv_num_heads=4,
    head_size=64,
):
    """Helper to create a minimal ONNX Attention node for testing (opset 24).

    ONNX Attention inputs (opset 24):
        0: Q (required)
        1: K (optional)
        2: V (optional)
        3: attn_mask (optional)
        4: past_key (optional)
        5: past_value (optional)
    """
    node = onnx.NodeProto()
    node.op_type = "Attention"
    node.name = "test_attention"
    node.input.extend(inputs)
    node.output.extend(outputs)

    num_heads_attr = AttributeProto()
    num_heads_attr.name = "num_heads"
    num_heads_attr.type = AttributeProto.INT
    num_heads_attr.i = num_heads
    node.attribute.append(num_heads_attr)

    kv_num_heads_attr = AttributeProto()
    kv_num_heads_attr.name = "kv_num_heads"
    kv_num_heads_attr.type = AttributeProto.INT
    kv_num_heads_attr.i = kv_num_heads
    node.attribute.append(kv_num_heads_attr)

    head_size_attr = AttributeProto()
    head_size_attr.name = "head_size"
    head_size_attr.type = AttributeProto.INT
    head_size_attr.i = head_size
    node.attribute.append(head_size_attr)

    return node


class TestFromOnnxAttention:
    def test_basic_conversion(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op)

        assert plugin_op.op_type == "AttentionPlugin"
        assert plugin_op.domain == "trt"
        assert plugin_op.name == "test_attention"
        assert list(plugin_op.input) == ["q", "k", "v", "", "", "", "", "", "", ""]
        assert list(plugin_op.output) == ["output", ""]

    def test_qkv_inputs_mapped(self):
        op = _make_attention_node(
            inputs=["query_tensor", "key_tensor", "value_tensor"],
            outputs=["attn_output"],
        )
        plugin_op = from_onnx_attention(op)

        assert plugin_op.input[0] == "query_tensor"
        assert plugin_op.input[1] == "key_tensor"
        assert plugin_op.input[2] == "value_tensor"

    def test_past_key_value_from_input4(self):
        op = _make_attention_node(
            inputs=["q", "k", "v", "", "past_key", "past_value"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op)

        # past_key (input[4]) used as past_key_value
        assert plugin_op.input[3] == "past_key"

    def test_past_key_value_empty_when_not_present(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op)

        assert plugin_op.input[3] == ""

    def test_attention_mask_from_input3(self):
        op = _make_attention_node(
            inputs=["q", "k", "v", "attn_mask_tensor"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op)

        assert plugin_op.input[7] == "attn_mask_tensor"

    def test_context_lengths_from_kwarg(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op, context_lengths="context_lens")

        assert plugin_op.input[4] == "context_lens"

    def test_rope_rotary_cos_sin_from_kwarg(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op, rope_rotary_cos_sin="rope_embeds")

        assert plugin_op.input[5] == "rope_embeds"

    def test_kvcache_start_index_from_kwarg(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op, kvcache_start_index="kv_start_idx")

        assert plugin_op.input[6] == "kv_start_idx"

    def test_attention_pos_id_from_kwarg(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op, attention_pos_id="pos_ids")

        assert plugin_op.input[8] == "pos_ids"

    def test_k_v_scale_quant_orig_from_kwarg(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op, k_v_scale_quant_orig="kv_scales")

        assert plugin_op.input[9] == "kv_scales"

    def test_enable_tree_attention_kwarg(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op, enable_tree_attention=1)

        attr_names = [a.name for a in plugin_op.attribute]
        assert "enable_tree_attention" in attr_names
        for attr in plugin_op.attribute:
            if attr.name == "enable_tree_attention":
                assert attr.i == 1

    def test_enable_fp8_kv_cache_kwarg(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op, enable_fp8_kv_cache=1)

        attr_names = [a.name for a in plugin_op.attribute]
        assert "enable_fp8_kv_cache" in attr_names
        for attr in plugin_op.attribute:
            if attr.name == "enable_fp8_kv_cache":
                assert attr.i == 1

    def test_sliding_window_size_kwarg(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op, sliding_window_size=128)

        attr_names = [a.name for a in plugin_op.attribute]
        assert "sliding_window_size" in attr_names
        for attr in plugin_op.attribute:
            if attr.name == "sliding_window_size":
                assert attr.i == 128

    def test_sliding_window_size_not_added_when_default(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op, sliding_window_size=-1)

        attr_names = [a.name for a in plugin_op.attribute]
        assert "sliding_window_size" not in attr_names

    def test_num_kv_heads_defaults_to_num_heads(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
            num_heads=8,
            kv_num_heads=8,
        )
        plugin_op = from_onnx_attention(op)

        for attr in plugin_op.attribute:
            if attr.name == "num_kv_heads":
                assert attr.i == 8

    def test_outputs_mapped(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["attention_out", "present_key", "present_value"],
        )
        plugin_op = from_onnx_attention(op)

        assert plugin_op.output[0] == "attention_out"
        assert plugin_op.output[1] == "present_key"

    def test_full_conversion_with_all_inputs(self):
        op = _make_attention_node(
            inputs=["q", "k", "v", "attn_mask", "past_key", "past_value"],
            outputs=["attention_out", "present_key", "present_value"],
            num_heads=16,
            kv_num_heads=8,
            head_size=128,
        )
        plugin_op = from_onnx_attention(
            op,
            enable_tree_attention=1,
            enable_fp8_kv_cache=1,
            sliding_window_size=256,
            context_lengths="ctx_lens",
            rope_rotary_cos_sin="rope",
            kvcache_start_index="kv_idx",
            attention_pos_id="pos",
            k_v_scale_quant_orig="scales",
        )

        assert plugin_op.input == [
            "q",
            "k",
            "v",
            "past_key",
            "ctx_lens",
            "rope",
            "kv_idx",
            "attn_mask",
            "pos",
            "scales",
        ]

        attr_dict = {a.name: a.i for a in plugin_op.attribute}
        assert attr_dict["num_q_heads"] == 16
        assert attr_dict["num_kv_heads"] == 8
        assert attr_dict["head_size"] == 128
        assert attr_dict["enable_tree_attention"] == 1
        assert attr_dict["enable_fp8_kv_cache"] == 1
        assert attr_dict["sliding_window_size"] == 256

        assert plugin_op.output == ["attention_out", "present_key"]
