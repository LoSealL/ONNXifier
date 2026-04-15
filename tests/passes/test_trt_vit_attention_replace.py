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

# pylint: disable=redefined-outer-name
import onnx
import pytest
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor,
    make_tensor_value_info,
)

from onnxifier import ONNXIFIER_IR_VERSION, OnnxGraph, PassManager


def _make_vit_attention_graph(
    total_seq_len: int = 197,
    num_heads: int = 12,
    head_size: int = 64,
    batch_size: int = 1,
    opset: int = 24,
) -> OnnxGraph:
    """Create a model with Attention node for Vision Transformer.

    ViT attention uses 4D shape [batch, num_heads, seq_len, head_size] for Q/K/V.

    Pattern:
        Attention(q, k, v)
            -> (output)
    """
    # Inputs in 4D layout [batch, num_heads, seq_len, head_size]
    q_info = make_tensor_value_info(
        "q", TensorProto.FLOAT16, [batch_size, num_heads, total_seq_len, head_size]
    )
    k_info = make_tensor_value_info(
        "k", TensorProto.FLOAT16, [batch_size, num_heads, total_seq_len, head_size]
    )
    v_info = make_tensor_value_info(
        "v", TensorProto.FLOAT16, [batch_size, num_heads, total_seq_len, head_size]
    )

    # Attention node - no KV cache, no RoPE
    attention_node = make_node(
        "Attention",
        ["q", "k", "v"],
        ["output"],
        name="vit_attention",
        q_num_heads=num_heads,
        kv_num_heads=num_heads,
    )

    # Output in 4D layout [batch, num_heads, seq_len, head_size]
    output_info = make_tensor_value_info(
        "output", TensorProto.FLOAT16, [batch_size, num_heads, total_seq_len, head_size]
    )

    # Build graph
    graph = make_graph(
        [attention_node],
        "vit_attention_graph",
        [q_info, k_info, v_info],
        [output_info],
    )

    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[make_operatorsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def _make_qwen_like_vit_attention_graph(
    seq_len: int = 16,
    hidden_size: int = 1024,
    num_heads: int = 16,
    opset: int = 24,
) -> OnnxGraph:
    """Create a qwen-like graph with model input shape [seq_len, hidden_size]."""
    input_info = make_tensor_value_info(
        "input", TensorProto.FLOAT16, [seq_len, hidden_size]
    )

    reshape_shape = make_node(
        "Constant",
        [],
        ["reshape_shape"],
        name="reshape_shape",
        value=make_tensor(
            "reshape_shape_value",
            TensorProto.INT64,
            [3],
            [1, seq_len, hidden_size],
        ),
    )

    reshape_to_attention_input = make_node(
        "Reshape",
        ["input", "reshape_shape"],
        ["qkv"],
        name="reshape_to_attention_input",
    )

    attention_node = make_node(
        "Attention",
        ["qkv", "qkv", "qkv"],
        ["output"],
        name="vit_attention",
        q_num_heads=num_heads,
        kv_num_heads=num_heads,
    )

    output_info = make_tensor_value_info(
        "output", TensorProto.FLOAT16, [1, seq_len, hidden_size]
    )

    graph = make_graph(
        [reshape_shape, reshape_to_attention_input, attention_node],
        "qwen_like_vit_attention_graph",
        [input_info],
        [output_info],
    )

    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[make_operatorsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_trt_vit_attention_replace_basic():
    """Test basic TRT ViTAttentionPlugin replacement."""
    graph = _make_vit_attention_graph()

    # Apply the pass
    graph = PassManager(["trt_vit_attention_replace"]).optimize(graph, strict=True)

    trt_opset = [
        i for i in graph.model.opset_import if i.domain == "trt" and i.version >= 1
    ]
    assert trt_opset, "TRT opset import should be added"

    # Check that Attention node was replaced with ViTAttentionPlugin
    attention_found = False
    plugin_found = False
    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type == "Attention":
            attention_found = True
        if node.op_type == "ViTAttentionPlugin":
            plugin_found = True
            # Verify required attributes are set
            attr_names = [a.name for a in node.attribute]
            assert "num_heads" in attr_names
            assert "head_size" in attr_names

    assert not attention_found, "Attention node should be replaced"
    assert plugin_found, "ViTAttentionPlugin node should be created"

    # Check that shared inputs were added
    input_names = [inp.name for inp in graph.input]
    assert "cu_seqlens" in input_names
    assert "max_seqlen_carrier" in input_names


def test_trt_vit_attention_replace_attributes():
    """Test that attributes are correctly transferred to ViTAttentionPlugin."""
    num_heads = 16
    head_size = 32
    graph = _make_vit_attention_graph(
        total_seq_len=197,
        num_heads=num_heads,
        head_size=head_size,
        batch_size=2,
    )

    # Apply the pass
    graph = PassManager(["trt_vit_attention_replace"]).optimize(graph, strict=True)

    # Check that attributes are correctly set
    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type == "ViTAttentionPlugin":
            attrs = {
                a.name: a.i for a in node.attribute if a.type == onnx.AttributeProto.INT
            }
            assert attrs["num_heads"] == num_heads, (
                f"Expected num_heads={num_heads}, got {attrs['num_heads']}"
            )
            assert attrs["head_size"] == head_size, (
                f"Expected head_size={head_size}, got {attrs['head_size']}"
            )
            break


def test_trt_vit_attention_replace_multiple_nodes():
    """Test replacement of multiple ViT Attention nodes."""
    # Create a model with 2 attention nodes
    seq_len = 197
    num_heads = 12
    head_size = 64

    q_info = make_tensor_value_info(
        "q", TensorProto.FLOAT16, [1, num_heads, seq_len, head_size]
    )
    k_info = make_tensor_value_info(
        "k", TensorProto.FLOAT16, [1, num_heads, seq_len, head_size]
    )
    v_info = make_tensor_value_info(
        "v", TensorProto.FLOAT16, [1, num_heads, seq_len, head_size]
    )

    # First attention layer
    attn1 = make_node(
        "Attention",
        ["q", "k", "v"],
        ["attn1_out"],
        name="vit_attention1",
        q_num_heads=num_heads,
        kv_num_heads=num_heads,
    )

    # Second attention layer
    attn2 = make_node(
        "Attention",
        ["attn1_out", "k", "v"],
        ["attn2_out"],
        name="vit_attention2",
        q_num_heads=num_heads,
        kv_num_heads=num_heads,
    )

    graph = make_graph(
        [attn1, attn2],
        "test_graph",
        [q_info, k_info, v_info],
        [
            make_tensor_value_info(
                "attn2_out", TensorProto.FLOAT16, [1, num_heads, seq_len, head_size]
            ),
        ],
    )

    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[make_operatorsetid("", 24)],
    )
    onnx.checker.check_model(model)

    graph_obj = OnnxGraph(model)
    graph_obj = PassManager(["trt_vit_attention_replace"]).optimize(
        graph_obj, strict=True
    )

    # Check that both Attention nodes were replaced
    plugin_count = 0
    attention_count = 0
    for n in graph_obj:
        node = graph_obj.nodes[n]["pb"]
        if node.op_type == "ViTAttentionPlugin":
            plugin_count += 1
        elif node.op_type == "Attention":
            attention_count += 1

    assert attention_count == 0, "No Attention nodes should remain"
    assert plugin_count == 2, f"Expected 2 ViTAttentionPlugin nodes, got {plugin_count}"

    # Check that shared inputs were added only once
    input_names = [inp.name for inp in graph_obj.input]
    assert input_names.count("cu_seqlens") == 1
    assert input_names.count("max_seqlen_carrier") == 1


def test_trt_vit_attention_replace_cu_seqlens_shape():
    """Test that cu_seqlens input has the correct shape."""
    graph = _make_vit_attention_graph(batch_size=3)

    # Apply the pass
    graph = PassManager(["trt_vit_attention_replace"]).optimize(graph, strict=True)

    # Check that cu_seqlens has shape [batch_size + 1]
    for inp in graph.input:
        if inp.name == "cu_seqlens":
            # cu_seqlens should be int32
            assert inp.type.tensor_type.elem_type == TensorProto.INT32
            # Shape should be [batch_size + 1] where batch_size is a symbolic dimension
            dims = inp.type.tensor_type.shape.dim
            assert len(dims) == 1
            assert dims[0].dim_value == 0 or dims[0].dim_param  # Either value or param
            break


def test_trt_vit_attention_replace_max_seqlen_shape():
    """Test that max_seqlen_carrier input is shape-only."""
    graph = _make_vit_attention_graph(batch_size=2)

    # Apply the pass
    graph = PassManager(["trt_vit_attention_replace"]).optimize(graph, strict=True)

    # Check that max_seqlen_carrier is int32 shape-only tensor
    for inp in graph.input:
        if inp.name == "max_seqlen_carrier":
            assert inp.type.tensor_type.elem_type == TensorProto.INT32
            # Should be 1-D
            dims = inp.type.tensor_type.shape.dim
            assert len(dims) == 1
            break


def test_trt_vit_attention_replace_input_mapping():
    """Test that Q/K/V inputs are correctly mapped."""
    graph = _make_vit_attention_graph()

    # Apply the pass
    graph = PassManager(["trt_vit_attention_replace"]).optimize(graph, strict=True)

    # Check the ViTAttentionPlugin node
    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type == "ViTAttentionPlugin":
            # Should have exactly 5 inputs: q, k, v, cu_seqlens, max_seqlen_carrier
            assert len(node.input) == 5, f"Expected 5 inputs, got {len(node.input)}"
            assert node.input[3] == "cu_seqlens"
            assert node.input[4] == "max_seqlen_carrier"
            # Should have exactly 1 output
            assert len(node.output) == 1
            break


def test_trt_vit_attention_replace_bake_seqlens_qwen_like_input():
    """Test baking cu_seqlens/max_seqlen_carrier from qwen-like [16, 1024] input."""
    graph = _make_qwen_like_vit_attention_graph(seq_len=16, hidden_size=1024)

    graph = PassManager(["trt_vit_attention_replace"]).optimize(graph, strict=True)

    plugin_node = None
    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type == "ViTAttentionPlugin":
            plugin_node = node
            break

    assert plugin_node is not None
    assert plugin_node.input[3] == "cu_seqlens_output_0"
    assert plugin_node.input[4] == "max_seqlen_carrier_output_0"

    input_names = [inp.name for inp in graph.input]
    assert "cu_seqlens" not in input_names
    assert "max_seqlen_carrier" not in input_names

    constant_values = {}
    constant_types = {}
    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type != "Constant":
            continue
        if node.name not in {"cu_seqlens", "max_seqlen_carrier"}:
            continue
        value_attr = next(attr for attr in node.attribute if attr.name == "value")
        constant_values[node.name] = onnx.numpy_helper.to_array(value_attr.t).tolist()
        constant_types[node.name] = value_attr.t.data_type

    assert constant_values["cu_seqlens"] == [0, 16]
    assert constant_values["max_seqlen_carrier"] == [16]
    assert constant_types["cu_seqlens"] == TensorProto.INT32
    assert constant_types["max_seqlen_carrier"] == TensorProto.INT32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
