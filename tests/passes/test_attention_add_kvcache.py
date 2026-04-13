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

import numpy as np
from onnx import numpy_helper
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier import PassManager
from onnxifier.graph import OnnxGraph


def _make_model_with_opset(opset_version: int):
    """Create a model with specified opset version."""

    def make_graph_with_attention():
        # Create an attention node without KV cache
        # Attention inputs (by position): q, k, v
        # Attention outputs: Y
        attention = make_node(
            "Attention",
            ["q", "k", "v"],
            ["output"],
            name="attention",
        )
        graph = make_graph(
            [attention],
            "graph",
            [
                make_value_info(
                    "q", make_tensor_type_proto(1, ["batch", "seq", "hidden"])
                ),
                make_value_info("k", make_tensor_type_proto(1, ["hidden", "hidden"])),
                make_value_info("v", make_tensor_type_proto(1, ["hidden"])),
            ],
            [make_value_info("output", make_tensor_type_proto(1, [1, 384, 768]))],
            [
                numpy_helper.from_array(
                    np.random.randn(768, 768).astype(np.float32), "k"
                ),
                numpy_helper.from_array(np.random.randn(768).astype(np.float32), "v"),
            ],
        )
        return graph

    graph_def = make_graph_with_attention()
    opset_imports = [make_operatorsetid("", opset_version)]
    return make_model(graph_def, opset_imports=opset_imports)


def _make_model_with_kv_cache():
    """Create a model with attention that already has KV cache."""

    def make_graph_with_attention_kv():
        # Attention with KV cache uses optional input positions:
        # [Q, K, V, attn_mask, past_key, past_value]
        # with empty attn_mask placeholder when no mask is provided.
        attention = make_node(
            "Attention",
            ["q", "k", "v", "", "past_key", "past_value"],
            ["output", "present_key", "present_value"],
            name="attention",
        )
        graph = make_graph(
            [attention],
            "graph",
            [
                make_value_info(
                    "q", make_tensor_type_proto(1, ["batch", "seq", "hidden"])
                ),
                make_value_info("k", make_tensor_type_proto(1, ["hidden", "hidden"])),
                make_value_info("v", make_tensor_type_proto(1, ["hidden"])),
                make_value_info(
                    "past_key",
                    make_tensor_type_proto(1, ["batch", "heads", "past_seq", "dim"]),
                ),
                make_value_info(
                    "past_value",
                    make_tensor_type_proto(1, ["batch", "heads", "past_seq", "dim"]),
                ),
            ],
            [
                make_value_info("output", make_tensor_type_proto(1, [1, 384, 768])),
                make_value_info(
                    "present_key",
                    make_tensor_type_proto(1, ["batch", "heads", "pres_seq", "dim"]),
                ),
                make_value_info(
                    "present_value",
                    make_tensor_type_proto(1, ["batch", "heads", "pres_seq", "dim"]),
                ),
            ],
            [
                numpy_helper.from_array(
                    np.random.randn(768, 768).astype(np.float32), "k"
                ),
                numpy_helper.from_array(np.random.randn(768).astype(np.float32), "v"),
            ],
        )
        return graph

    graph_def = make_graph_with_attention_kv()
    opset_imports = [make_operatorsetid("", 23)]
    return make_model(graph_def, opset_imports=opset_imports)


def _make_model_with_wrong_op_type():
    """Create a model with a non-attention op."""

    def make_graph_with_matmul():
        matmul = make_node(
            "MatMul",
            ["input_a", "input_b"],
            ["output"],
            name="matmul",
        )
        graph = make_graph(
            [matmul],
            "graph",
            [
                make_value_info("input_a", make_tensor_type_proto(1, [10, 20])),
                make_value_info("input_b", make_tensor_type_proto(1, [20, 30])),
            ],
            [make_value_info("output", make_tensor_type_proto(1, [10, 30]))],
            [],
        )
        return graph

    graph_def = make_graph_with_matmul()
    opset_imports = [make_operatorsetid("", 23)]
    return make_model(graph_def, opset_imports=opset_imports)


def _make_model_with_4d_input():
    """Create a model with attention with 4D input tensor."""

    def make_graph_with_4d_input():
        attention = make_node(
            "Attention",
            ["q", "k", "v"],
            ["output"],
            name="attention",
        )
        graph = make_graph(
            [attention],
            "graph",
            [
                make_value_info(
                    "q",
                    make_tensor_type_proto(1, ["batch", "heads", "seq", "dim"]),
                ),
                make_value_info("k", make_tensor_type_proto(1, ["hidden", "hidden"])),
                make_value_info("v", make_tensor_type_proto(1, ["hidden"])),
            ],
            [
                make_value_info(
                    "output",
                    make_tensor_type_proto(1, ["batch", "heads", "seq", "dim"]),
                )
            ],
            [
                numpy_helper.from_array(
                    np.random.randn(768, 768).astype(np.float32), "k"
                ),
                numpy_helper.from_array(np.random.randn(768).astype(np.float32), "v"),
            ],
        )
        return graph

    graph_def = make_graph_with_4d_input()
    opset_imports = [make_operatorsetid("", 24)]
    return make_model(graph_def, opset_imports=opset_imports)


def _make_model_with_unknown_input_shape():
    """Create a model with attention with unknown input shape."""

    def make_graph_with_unknown_shape():
        attention = make_node(
            "Attention",
            ["q", "k", "v"],
            ["output"],
            name="attention",
        )
        graph = make_graph(
            [attention],
            "graph",
            [
                make_value_info("q", make_tensor_type_proto(1, None)),
                make_value_info("k", make_tensor_type_proto(1, ["hidden", "hidden"])),
                make_value_info("v", make_tensor_type_proto(1, ["hidden"])),
            ],
            [make_value_info("output", make_tensor_type_proto(1, None))],
            [
                numpy_helper.from_array(
                    np.random.randn(768, 768).astype(np.float32), "k"
                ),
                numpy_helper.from_array(np.random.randn(768).astype(np.float32), "v"),
            ],
        )
        return graph

    graph_def = make_graph_with_unknown_shape()
    opset_imports = [make_operatorsetid("", 23)]
    return make_model(graph_def, opset_imports=opset_imports)


class TestAttentionAddKVCache:
    """Test cases for attention_add_kvcache rewriter."""

    def test_opset23_adds_kvcache(self):
        """Test that KV cache is added for opset 23."""
        model = _make_model_with_opset(23)
        graph = OnnxGraph(model)
        assert graph.opset_version == 23

        pm = PassManager(["attention_add_kvcache"])
        graph = pm.optimize(graph)

        # Check that graph inputs include past_key and past_value
        input_names = [inp.name for inp in graph.input]
        assert any("past_key" in name for name in input_names)
        assert any("past_value" in name for name in input_names)

        # Check that graph outputs include present_key and present_value
        output_names = [out.name for out in graph.output]
        assert any("present_key" in name for name in output_names)
        assert any("present_value" in name for name in output_names)

        # Check the attention node has 6 inputs with optional attn_mask slot kept empty
        attention_node = graph.nodes["attention"]["pb"]
        assert len(attention_node.input) == 6
        assert attention_node.input[3] == ""
        assert attention_node.input[4] == "attention/past_key"
        assert attention_node.input[5] == "attention/past_value"

        # Check the attention node has 3 outputs
        assert len(attention_node.output) == 3
        assert attention_node.output[1] == "attention/present_key"
        assert attention_node.output[2] == "attention/present_value"

        # Check past/present sequence lengths are dynamic dims
        past_key_input = next(
            inp for inp in graph.input if inp.name == "attention/past_key"
        )
        present_key_output = next(
            out for out in graph.output if out.name == "attention/present_key"
        )
        assert past_key_input.type.tensor_type.shape.dim[2].dim_param == "past_seq_len"
        assert (
            present_key_output.type.tensor_type.shape.dim[2].dim_param
            == "present_seq_len"
        )

    def test_opset24_adds_kvcache(self):
        """Test that KV cache is added for opset 24."""
        model = _make_model_with_opset(24)
        graph = OnnxGraph(model)
        assert graph.opset_version == 24

        pm = PassManager(["attention_add_kvcache"])
        graph = pm.optimize(graph)

        # Check that graph inputs include past_key and past_value
        input_names = [inp.name for inp in graph.input]
        assert any("past_key" in name for name in input_names)
        assert any("past_value" in name for name in input_names)

        # Check that graph outputs include present_key and present_value
        output_names = [out.name for out in graph.output]
        assert any("present_key" in name for name in output_names)
        assert any("present_value" in name for name in output_names)

    def test_skips_opset18(self):
        """Test that KV cache is NOT added for opset 18."""
        model = _make_model_with_opset(18)
        graph = OnnxGraph(model)
        assert graph.opset_version == 18

        original_input_count = len(graph.input)
        original_output_count = len(graph.output)

        pm = PassManager(["attention_add_kvcache"])
        graph = pm.optimize(graph)

        # Should not add new inputs
        assert len(graph.input) == original_input_count
        # Should not add new outputs
        assert len(graph.output) == original_output_count

    def test_skips_already_has_kvcache(self):
        """Test that KV cache is NOT added if attention already has it."""
        model = _make_model_with_kv_cache()
        graph = OnnxGraph(model)
        assert graph.opset_version == 23

        original_input_count = len(graph.input)
        original_output_count = len(graph.output)

        pm = PassManager(["attention_add_kvcache"])
        graph = pm.optimize(graph)

        # Should not add new inputs (already has 5)
        assert len(graph.input) == original_input_count
        # Should not add new outputs (already has 3)
        assert len(graph.output) == original_output_count

    def test_skips_non_attention_op(self):
        """Test that non-attention ops are not modified."""
        model = _make_model_with_wrong_op_type()
        graph = OnnxGraph(model)
        assert graph.opset_version == 23

        original_input_count = len(graph.input)
        original_output_count = len(graph.output)

        pm = PassManager(["attention_add_kvcache"])
        graph = pm.optimize(graph)

        # Should not add new inputs
        assert len(graph.input) == original_input_count
        # Should not add new outputs
        assert len(graph.output) == original_output_count

    def test_with_4d_input_shape(self):
        """Test with 4D input tensor [batch, heads, seq, dim]."""
        model = _make_model_with_4d_input()
        graph = OnnxGraph(model)
        assert graph.opset_version == 24

        pm = PassManager(["attention_add_kvcache"])
        graph = pm.optimize(graph)

        # Check that graph inputs include past_key and past_value
        input_names = [inp.name for inp in graph.input]
        assert any("past_key" in name for name in input_names)
        assert any("past_value" in name for name in input_names)

        # Check that graph outputs include present_key and present_value
        output_names = [out.name for out in graph.output]
        assert any("present_key" in name for name in output_names)
        assert any("present_value" in name for name in output_names)

    def test_with_unknown_input_shape(self):
        """Test with unknown input shape (None)."""
        model = _make_model_with_unknown_input_shape()
        graph = OnnxGraph(model)
        assert graph.opset_version == 23

        pm = PassManager(["attention_add_kvcache"])
        graph = pm.optimize(graph)

        # Should still work with unknown shapes
        input_names = [inp.name for inp in graph.input]
        assert any("past_key" in name for name in input_names)
        assert any("past_value" in name for name in input_names)

        output_names = [out.name for out in graph.output]
        assert any("present_key" in name for name in output_names)
        assert any("present_value" in name for name in output_names)

    def test_with_sequence_length_param(self):
        """Test with explicit sequence_length parameter."""
        from onnxifier.passes.swap.attention_add_kvcache import (
            AttentionAddKVCacheRewriter,
        )

        model = _make_model_with_opset(23)
        graph = OnnxGraph(model)
        assert graph.opset_version == 23

        rewriter = AttentionAddKVCacheRewriter()
        graph = rewriter(graph)

        # Check that the rewriter was applied
        input_names = [inp.name for inp in graph.input]
        assert any("past_key" in name for name in input_names)

        # Sequence length dims remain dynamic even when sequence_length is provided
        past_key_input = next(
            inp for inp in graph.input if inp.name == "attention/past_key"
        )
        present_key_output = next(
            out for out in graph.output if out.name == "attention/present_key"
        )
        assert past_key_input.type.tensor_type.shape.dim[2].dim_param == "past_seq_len"
        assert (
            present_key_output.type.tensor_type.shape.dim[2].dim_param
            == "present_seq_len"
        )

    def test_helper_make_past_shape(self):
        """Test _make_past_shape helper method."""
        from onnxifier.passes.swap.attention_add_kvcache import (
            AttentionAddKVCacheRewriter,
        )

        rewriter = AttentionAddKVCacheRewriter()

        # Test with None input shape - returns placeholder strings for dynamic shape
        shape = rewriter._make_past_shape(None, 0)
        assert shape == ["batch_size", "num_heads", "past_seq_len", "head_dim"]

        # Test with 4D input shape
        input_shape_4d: list[int | str] = ["batch", 12, "seq", 64]
        shape = rewriter._make_past_shape(input_shape_4d, 10)
        assert shape == ["batch", 12, 10, 64]

        # Test with 3D input shape
        input_shape_3d: list[int | str] = ["batch", "seq", 768]
        shape = rewriter._make_past_shape(input_shape_3d, 5)
        assert shape == ["batch_size", "num_heads", 5, "head_dim"]

        # Test with other input shape
        input_shape_other: list[int | str] = ["batch", "seq"]
        shape = rewriter._make_past_shape(input_shape_other, 3)
        assert shape == ["batch_size", "num_heads", 3, "head_dim"]

    def test_helper_make_present_shape(self):
        """Test _make_present_shape helper method."""
        from onnxifier.passes.swap.attention_add_kvcache import (
            AttentionAddKVCacheRewriter,
        )

        rewriter = AttentionAddKVCacheRewriter()

        # Test with None input shape - returns placeholder strings for dynamic shape
        shape = rewriter._make_present_shape(None, 0)
        assert shape == ["batch_size", "num_heads", "present_seq_len", "head_dim"]

        # Test with 4D input shape
        input_shape_4d: list[int | str] = ["batch", 12, "seq", 64]
        shape = rewriter._make_present_shape(input_shape_4d, 10)
        assert shape == ["batch", 12, 10, 64]

        # Test with 3D input shape
        input_shape_3d: list[int | str] = ["batch", "seq", 768]
        shape = rewriter._make_present_shape(input_shape_3d, 5)
        assert shape == ["batch_size", "num_heads", 5, "head_dim"]

        # Test with other input shape
        input_shape_other: list[int | str] = ["batch", "seq"]
        shape = rewriter._make_present_shape(input_shape_other, 3)
        assert shape == ["batch_size", "num_heads", 3, "head_dim"]

    def test_graph_outputs_updated(self):
        """Test that graph outputs dict is properly updated."""
        model = _make_model_with_opset(23)
        graph = OnnxGraph(model)
        assert graph.opset_version == 23

        pm = PassManager(["attention_add_kvcache"])
        graph = pm.optimize(graph)

        # Check outputs dict is updated
        assert "attention/present_key" in graph.outputs
        assert "attention/present_value" in graph.outputs

    def test_graph_inputs_updated(self):
        """Test that graph inputs dict is properly updated."""
        model = _make_model_with_opset(23)
        graph = OnnxGraph(model)
        assert graph.opset_version == 23

        pm = PassManager(["attention_add_kvcache"])
        graph = pm.optimize(graph)

        # Check inputs dict is updated
        assert "attention/past_key" in graph.inputs
        assert "attention/past_value" in graph.inputs

    def test_onnx_model_round_trip(self):
        """Test that the modified graph can be converted to onnx model."""
        model = _make_model_with_opset(23)
        graph = OnnxGraph(model)
        assert graph.opset_version == 23

        pm = PassManager(["attention_add_kvcache"])
        graph = pm.optimize(graph)

        # Should be able to get model without error
        onnx_model = graph.model
        assert onnx_model is not None
        assert len(onnx_model.graph.node) == 1

        attention_node = onnx_model.graph.node[0]
        assert len(attention_node.input) == 6
        assert len(attention_node.output) == 3

    def test_multiple_attention_nodes(self):
        """Test with multiple attention nodes in the graph."""

        def make_graph_with_two_attention():
            attention1 = make_node(
                "Attention",
                ["q1", "k1", "v1"],
                ["output1"],
                name="attention1",
            )
            attention2 = make_node(
                "Attention",
                ["q2", "k2", "v2"],
                ["output2"],
                name="attention2",
            )
            graph = make_graph(
                [attention1, attention2],
                "graph",
                [
                    make_value_info(
                        "q1",
                        make_tensor_type_proto(1, ["batch", "seq", "hidden"]),
                    ),
                    make_value_info(
                        "k1", make_tensor_type_proto(1, ["hidden", "hidden"])
                    ),
                    make_value_info("v1", make_tensor_type_proto(1, ["hidden"])),
                    make_value_info(
                        "q2",
                        make_tensor_type_proto(1, ["batch", "seq", "hidden"]),
                    ),
                    make_value_info(
                        "k2", make_tensor_type_proto(1, ["hidden", "hidden"])
                    ),
                    make_value_info("v2", make_tensor_type_proto(1, ["hidden"])),
                ],
                [
                    make_value_info(
                        "output1", make_tensor_type_proto(1, [1, 384, 768])
                    ),
                    make_value_info(
                        "output2", make_tensor_type_proto(1, [1, 384, 768])
                    ),
                ],
                [
                    numpy_helper.from_array(
                        np.random.randn(768, 768).astype(np.float32), "k1"
                    ),
                    numpy_helper.from_array(
                        np.random.randn(768).astype(np.float32), "v1"
                    ),
                    numpy_helper.from_array(
                        np.random.randn(768, 768).astype(np.float32), "k2"
                    ),
                    numpy_helper.from_array(
                        np.random.randn(768).astype(np.float32), "v2"
                    ),
                ],
            )
            return graph

        graph_def = make_graph_with_two_attention()
        opset_imports = [make_operatorsetid("", 23)]
        model = make_model(graph_def, opset_imports=opset_imports)

        graph = OnnxGraph(model)
        assert graph.opset_version == 23

        pm = PassManager(["attention_add_kvcache"])
        graph = pm.optimize(graph)

        # Both attention nodes should have KV cache added
        attention1 = graph.nodes["attention1"]["pb"]
        assert len(attention1.input) == 6
        assert len(attention1.output) == 3

        attention2 = graph.nodes["attention2"]["pb"]
        assert len(attention2.input) == 6
        assert len(attention2.output) == 3

    def test_empty_past_key_value(self):
        """Test attention with empty past_key and past_value inputs.

        When optional placeholders exist, the rewriter fills past_key/past_value at
        canonical positions 4 and 5.
        """

        def make_graph_with_empty_past():
            attention = make_node(
                "Attention",
                ["q", "k", "v", "", "", ""],
                ["output"],
                name="attention",
            )
            graph = make_graph(
                [attention],
                "graph",
                [
                    make_value_info(
                        "q",
                        make_tensor_type_proto(1, ["batch", "seq", "hidden"]),
                    ),
                    make_value_info(
                        "k", make_tensor_type_proto(1, ["hidden", "hidden"])
                    ),
                    make_value_info("v", make_tensor_type_proto(1, ["hidden"])),
                ],
                [make_value_info("output", make_tensor_type_proto(1, [1, 384, 768]))],
                [
                    numpy_helper.from_array(
                        np.random.randn(768, 768).astype(np.float32), "k"
                    ),
                    numpy_helper.from_array(
                        np.random.randn(768).astype(np.float32), "v"
                    ),
                ],
            )
            return graph

        graph_def = make_graph_with_empty_past()
        opset_imports = [make_operatorsetid("", 23)]
        model = make_model(graph_def, opset_imports=opset_imports)

        graph = OnnxGraph(model)
        assert graph.opset_version == 23

        pm = PassManager(["attention_add_kvcache"])
        graph = pm.optimize(graph)

        attention = graph.nodes["attention"]["pb"]
        assert len(attention.input) == 6
        assert attention.input[4] == "attention/past_key"
        assert attention.input[5] == "attention/past_value"
