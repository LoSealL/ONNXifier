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

from onnxifier.domain.trt.ops.vit_attention_plugin import from_onnx_attention


def _make_attention_node(
    inputs,
    outputs,
    num_heads=4,
    head_size=64,
):
    """Helper to create a minimal ONNX Attention node for testing (opset 24)."""
    node = onnx.NodeProto()
    node.op_type = "Attention"
    node.name = "test_attention"
    node.input.extend(inputs)
    node.output.extend(outputs)

    num_heads_attr = AttributeProto()
    num_heads_attr.name = "q_num_heads"
    num_heads_attr.type = AttributeProto.INT
    num_heads_attr.i = num_heads
    node.attribute.append(num_heads_attr)

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
        plugin_op = from_onnx_attention(op, 64)

        assert plugin_op.op_type == "ViTAttentionPlugin"
        assert plugin_op.domain == "trt"
        assert plugin_op.name == "test_attention"
        assert list(plugin_op.input) == ["q", "k", "v", "", ""]
        assert list(plugin_op.output) == ["output"]

    def test_qkv_inputs_mapped(self):
        op = _make_attention_node(
            inputs=["query_tensor", "key_tensor", "value_tensor"],
            outputs=["attn_output"],
        )
        plugin_op = from_onnx_attention(op, 64)

        assert plugin_op.input[0] == "query_tensor"
        assert plugin_op.input[1] == "key_tensor"
        assert plugin_op.input[2] == "value_tensor"

    def test_cu_seqlens_from_kwarg(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op, 64, cu_seqlens="cu_seqlens_tensor")

        assert plugin_op.input[3] == "cu_seqlens_tensor"

    def test_max_seqlen_carrier_from_kwarg(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
        )
        plugin_op = from_onnx_attention(op, 64, max_seqlen_carrier="max_seqlen_tensor")

        assert plugin_op.input[4] == "max_seqlen_tensor"

    def test_num_heads_attribute(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
            num_heads=8,
        )
        plugin_op = from_onnx_attention(op, 64)

        for attr in plugin_op.attribute:
            if attr.name == "num_heads":
                assert attr.i == 8

    def test_head_size_attribute(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["output"],
            head_size=128,
        )
        plugin_op = from_onnx_attention(op, 128)

        for attr in plugin_op.attribute:
            if attr.name == "head_size":
                assert attr.i == 128

    def test_output_mapped(self):
        op = _make_attention_node(
            inputs=["q", "k", "v"],
            outputs=["attention_out", "present_key", "present_value"],
        )
        plugin_op = from_onnx_attention(op, 64)

        assert plugin_op.output[0] == "attention_out"

    def test_full_conversion(self):
        op = _make_attention_node(
            inputs=["q", "k", "v", "attn_mask", "past_key", "past_value"],
            outputs=["attention_out", "present_key", "present_value"],
            num_heads=12,
            head_size=64,
        )
        plugin_op = from_onnx_attention(
            op,
            64,
            cu_seqlens="seq_lengths",
            max_seqlen_carrier="max_seq",
        )

        assert plugin_op.input == ["q", "k", "v", "seq_lengths", "max_seq"]
        assert plugin_op.output == ["attention_out"]

        attr_dict = {a.name: a.i for a in plugin_op.attribute}
        assert attr_dict["num_heads"] == 12
        assert attr_dict["head_size"] == 64
