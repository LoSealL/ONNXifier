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
import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, OnnxGraph, PassManager


def _make_attention_with_kvcache_graph(
    batch_size: int = 2,
    seq_len: int = 10,
    num_q_heads: int = 4,
    num_kv_heads: int = 2,
    head_size: int = 16,
    has_attention_mask: bool = False,
    opset: int = 24,
) -> OnnxGraph:
    """Create a model with Attention node including KV cache.

    Pattern:
        Attention(q, k, v, attn_mask, past_key, past_value)
            -> (output, present_key, present_value)
    """
    # Inputs with 4D shapes for opset 24:
    # [batch, num_heads, seq_len, head_size]
    q_info = make_tensor_value_info(
        "q", TensorProto.FLOAT16, [batch_size, num_q_heads, seq_len, head_size]
    )
    k_info = make_tensor_value_info(
        "k", TensorProto.FLOAT16, [batch_size, num_kv_heads, seq_len, head_size]
    )
    v_info = make_tensor_value_info(
        "v", TensorProto.FLOAT16, [batch_size, num_kv_heads, seq_len, head_size]
    )
    past_key_info = make_tensor_value_info(
        "past_key",
        TensorProto.FLOAT16,
        [batch_size, num_kv_heads, 0, head_size],
    )
    past_value_info = make_tensor_value_info(
        "past_value",
        TensorProto.FLOAT16,
        [batch_size, num_kv_heads, 0, head_size],
    )

    # Build attention inputs
    attention_inputs = ["q", "k", "v"]

    if has_attention_mask:
        attention_inputs.append("attn_mask")
    else:
        attention_inputs.append("")

    attention_inputs.extend(["past_key", "past_value"])

    # Attention node with proper attributes for opset 24
    # For 4D input shapes, the pass will infer num_heads and head_size
    attention_node = make_node(
        "Attention",
        attention_inputs,
        ["output", "present_key", "present_value"],
        name="attention",
        q_num_heads=num_q_heads,
        kv_num_heads=num_kv_heads,
    )

    # Build initializers
    initializers = []
    if has_attention_mask:
        attn_mask = from_array(
            np.zeros((seq_len, seq_len), dtype=np.int32), "attn_mask"
        )
        initializers.append(attn_mask)

    # Build graph
    graph_inputs = [q_info, k_info, v_info, past_key_info, past_value_info]
    if has_attention_mask:
        graph_inputs.insert(
            3,
            make_tensor_value_info("attn_mask", TensorProto.INT32, [seq_len, seq_len]),
        )

    graph = make_graph(
        [attention_node],
        "attention_graph",
        graph_inputs,
        [
            make_tensor_value_info(
                "output",
                TensorProto.FLOAT16,
                [batch_size, seq_len, num_q_heads * head_size],
            ),
            make_tensor_value_info(
                "present_key",
                TensorProto.FLOAT16,
                [batch_size, num_kv_heads, seq_len, head_size],
            ),
            make_tensor_value_info(
                "present_value",
                TensorProto.FLOAT16,
                [batch_size, num_kv_heads, seq_len, head_size],
            ),
        ],
        initializer=initializers,
    )

    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[make_operatorsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_trt_attention_replace_basic():
    """Test basic TRT AttentionPlugin replacement."""
    graph = _make_attention_with_kvcache_graph()

    # Apply the pass
    graph = PassManager(["trt_attention_replace"]).optimize(graph, strict=True)

    trt_opset = [
        i for i in graph.model.opset_import if i.domain == "trt" and i.version >= 1
    ]
    assert trt_opset, "TRT opset import should be added"

    # Check that Attention node was replaced with AttentionPlugin
    attention_found = False
    plugin_found = False
    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type == "Attention":
            attention_found = True
        if node.op_type == "AttentionPlugin":
            plugin_found = True
            # Verify required attributes are set
            attr_names = [a.name for a in node.attribute]
            assert "num_q_heads" in attr_names
            assert "num_kv_heads" in attr_names
            assert "head_size" in attr_names
            assert "enable_tree_attention" in attr_names

    assert not attention_found, "Attention node should be replaced"
    assert plugin_found, "AttentionPlugin node should be created"

    # Check that shared inputs were added
    input_names = [inp.name for inp in graph.input]
    assert "context_lengths" in input_names
    assert "rope_rotary_cos_sin" in input_names
    assert "kvcache_start_index" in input_names

    # Split KV cache IO should be replaced by merged IO
    assert "past_key" not in input_names
    assert "past_value" not in input_names
    merged_past = "past_key_value"
    assert merged_past in input_names

    output_names = [out.name for out in graph.output]
    assert "present_key" not in output_names
    assert "present_value" not in output_names
    merged_present = "present_key_value"
    assert merged_present in output_names

    # merged shape is concat([key, value], axis=1)
    merged_past_shape = graph.tensor_shape(merged_past)
    assert merged_past_shape == [2, 2, 2, 0, 16]

    merged_present_shape = graph.tensor_shape(merged_present)
    assert merged_present_shape == [2, 2, 2, 10, 16]

    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type == "AttentionPlugin":
            assert node.input[3] == merged_past
            assert node.output[1] == merged_present
            break


def test_trt_attention_replace_with_mask():
    """Test TRT AttentionPlugin replacement with attention mask."""
    graph = _make_attention_with_kvcache_graph(has_attention_mask=True)

    # Apply the pass
    graph = PassManager(["trt_attention_replace"]).optimize(graph, strict=True)

    # Check that the attention mask is passed to AttentionPlugin
    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type == "AttentionPlugin":
            # AttentionPlugin should have inputs for attention_mask
            assert (
                len(node.input) >= 8
            )  # q, k, v, past_kv, context_len, rope, cache_idx, mask
            # The attention mask input should be at index 7 or later
            break


def test_trt_attention_replace_keeps_shape_inference_for_downstream_ops():
    """Test that downstream infer_shape still works after AttentionPlugin swap."""
    batch_size = 2
    seq_len = 10
    num_heads = 4
    head_size = 16

    q_info = make_tensor_value_info(
        "q", TensorProto.FLOAT16, [batch_size, num_heads, seq_len, head_size]
    )
    k_info = make_tensor_value_info(
        "k", TensorProto.FLOAT16, [batch_size, num_heads, seq_len, head_size]
    )
    v_info = make_tensor_value_info(
        "v", TensorProto.FLOAT16, [batch_size, num_heads, seq_len, head_size]
    )
    pk_info = make_tensor_value_info(
        "past_key", TensorProto.FLOAT16, [batch_size, num_heads, 0, head_size]
    )
    pv_info = make_tensor_value_info(
        "past_value", TensorProto.FLOAT16, [batch_size, num_heads, 0, head_size]
    )

    attn = make_node(
        "Attention",
        ["q", "k", "v", "", "past_key", "past_value"],
        ["attn_out", "present_key", "present_value"],
        name="attention",
        q_num_heads=num_heads,
        kv_num_heads=num_heads,
    )
    # If AttentionPlugin output keeps value_info, infer_shape can infer this output.
    transpose = make_node(
        "Transpose",
        ["attn_out"],
        ["attn_out_t"],
        name="transpose_after_attention",
        perm=[0, 2, 1],
    )
    identity = make_node(
        "Identity",
        ["attn_out_t"],
        ["attn_out_final"],
        name="identity_after_transpose",
    )

    graph = make_graph(
        [attn, transpose, identity],
        "shape_inference_graph",
        [q_info, k_info, v_info, pk_info, pv_info],
        [
            make_tensor_value_info(
                "attn_out_final",
                TensorProto.FLOAT16,
                [batch_size, num_heads * head_size, seq_len],
            ),
            make_tensor_value_info(
                "present_key",
                TensorProto.FLOAT16,
                [batch_size, num_heads, seq_len, head_size],
            ),
            make_tensor_value_info(
                "present_value",
                TensorProto.FLOAT16,
                [batch_size, num_heads, seq_len, head_size],
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
    graph_obj = PassManager(["trt_attention_replace", "infer_shape"]).optimize(
        graph_obj, strict=True
    )

    # AttentionPlugin output should keep shape info for downstream inference.
    assert graph_obj.tensor_shape("attn_out") == [
        batch_size,
        num_heads,
        seq_len,
        head_size,
    ]


def test_trt_attention_replace_multiple_nodes():
    """Test replacement of multiple Attention nodes."""
    # Create a model with 2 attention nodes
    batch_size = 2
    seq_len = 10
    num_heads = 4
    head_size = 16

    q_info = make_tensor_value_info(
        "q", TensorProto.FLOAT16, [batch_size, num_heads, seq_len, head_size]
    )
    k_info = make_tensor_value_info(
        "k", TensorProto.FLOAT16, [batch_size, num_heads, seq_len, head_size]
    )
    v_info = make_tensor_value_info(
        "v", TensorProto.FLOAT16, [batch_size, num_heads, seq_len, head_size]
    )
    pk_info = make_tensor_value_info(
        "past_key_0", TensorProto.FLOAT16, [batch_size, num_heads, 0, head_size]
    )
    pv_info = make_tensor_value_info(
        "past_value_0", TensorProto.FLOAT16, [batch_size, num_heads, 0, head_size]
    )

    # First attention layer
    attn1 = make_node(
        "Attention",
        ["q", "k", "v", "", "past_key_0", "past_value_0"],
        ["attn1_out", "present_key_0", "present_value_0"],
        name="attention1",
        q_num_heads=num_heads,
        kv_num_heads=num_heads,
    )

    # Second attention layer
    attn2 = make_node(
        "Attention",
        ["attn1_out", "k", "v", "", "past_key_0", "past_value_0"],
        ["attn2_out", "present_key_1", "present_value_1"],
        name="attention2",
        q_num_heads=num_heads,
        kv_num_heads=num_heads,
    )

    graph = make_graph(
        [attn1, attn2],
        "test_graph",
        [q_info, k_info, v_info, pk_info, pv_info],
        [
            make_tensor_value_info(
                "attn2_out",
                TensorProto.FLOAT16,
                [batch_size, num_heads, seq_len, head_size],
            ),
            make_tensor_value_info(
                "present_key_0",
                TensorProto.FLOAT16,
                [batch_size, num_heads, seq_len, head_size],
            ),
            make_tensor_value_info(
                "present_value_0",
                TensorProto.FLOAT16,
                [batch_size, num_heads, seq_len, head_size],
            ),
            make_tensor_value_info(
                "present_key_1",
                TensorProto.FLOAT16,
                [batch_size, num_heads, seq_len, head_size],
            ),
            make_tensor_value_info(
                "present_value_1",
                TensorProto.FLOAT16,
                [batch_size, num_heads, seq_len, head_size],
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
    graph_obj = PassManager(["trt_attention_replace"]).optimize(graph_obj, strict=True)

    trt_opset = [
        i for i in graph_obj.model.opset_import if i.domain == "trt" and i.version >= 1
    ]
    assert trt_opset, "TRT opset import should be added"

    # Check that both Attention nodes were replaced
    plugin_count = 0
    attention_count = 0
    for n in graph_obj:
        node = graph_obj.nodes[n]["pb"]
        if node.op_type == "AttentionPlugin":
            plugin_count += 1
        elif node.op_type == "Attention":
            attention_count += 1

    assert attention_count == 0, "No Attention nodes should remain"
    assert plugin_count == 2, f"Expected 2 AttentionPlugin nodes, got {plugin_count}"

    # Check that shared inputs were added only once
    input_names = [inp.name for inp in graph_obj.input]
    assert input_names.count("context_lengths") == 1
    assert input_names.count("rope_rotary_cos_sin") == 1
    assert input_names.count("kvcache_start_index") == 1

    # Split KV cache inputs should be removed and replaced by merged ones
    assert "past_key_0" not in input_names
    assert "past_value_0" not in input_names
    assert "past_key_1" not in input_names
    assert "past_value_1" not in input_names
    merged_inputs = [n for n in input_names if "past_key_value" in n]
    assert len(merged_inputs) == 1

    # Split KV cache outputs should be removed and replaced by merged ones
    output_names = [out.name for out in graph_obj.output]
    assert "present_key_0" not in output_names
    assert "present_value_0" not in output_names
    assert "present_key_1" not in output_names
    assert "present_value_1" not in output_names
    merged_outputs = [n for n in output_names if "present_key_value" in n]
    assert len(merged_outputs) == 2


def test_trt_attention_replace_with_parameters():
    """Test TRT AttentionPlugin replacement with custom parameters."""
    graph = _make_attention_with_kvcache_graph()

    # Apply the pass with custom parameters
    pass_manager = PassManager(
        ["trt_attention_replace"],
        configs={
            "trt_attention_replace": {
                "enable_tree_attention": 1,
                "enable_fp8_kv_cache": 0,
                "sliding_window_size": 4096,
            }
        },
    )
    graph = pass_manager.optimize(graph, strict=True)

    # Check that the parameters were applied
    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type == "AttentionPlugin":
            attrs = {
                a.name: a.i for a in node.attribute if a.type == onnx.AttributeProto.INT
            }
            assert attrs.get("enable_tree_attention") == 1
            assert attrs.get("sliding_window_size") == 4096
            break
