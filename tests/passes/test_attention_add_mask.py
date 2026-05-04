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
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array

from onnxifier import PassManager
from onnxifier.graph import OnnxGraph


def _make_model_with_one_attention(
    opset: int = 23,
    has_mask: bool = False,
    mask_as_graph_input: bool = False,
):
    """Create a model with one Attention node."""
    attention_inputs = ["q", "k", "v"]
    if has_mask:
        attention_inputs.append("attn_mask")
    else:
        attention_inputs.append("")

    attention = make_node(
        "Attention",
        attention_inputs,
        ["output"],
        name="attention",
    )

    graph_inputs = [
        make_tensor_value_info("q", TensorProto.FLOAT, [1, 16, 256]),
        make_tensor_value_info("k", TensorProto.FLOAT, [1, 16, 256]),
        make_tensor_value_info("v", TensorProto.FLOAT, [1, 16, 256]),
    ]
    initializers = [
        from_array(np.random.randn(1, 16, 256).astype(np.float32), "k"),
        from_array(np.random.randn(1, 16, 256).astype(np.float32), "v"),
    ]

    if mask_as_graph_input:
        graph_inputs.append(
            make_tensor_value_info("attn_mask", TensorProto.INT32, [16, 16])
        )

    graph = make_graph(
        [attention],
        "graph",
        graph_inputs,
        [make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 256])],
        initializers,
    )

    opset_imports = [make_operatorsetid("", opset)]
    return make_model(graph, opset_imports=opset_imports)


def _make_model_with_two_attention(opset: int = 23):
    """Create a model with two Attention nodes without mask."""
    attention1 = make_node(
        "Attention",
        ["q", "k", "v", ""],
        ["output1"],
        name="attention1",
    )
    attention2 = make_node(
        "Attention",
        ["output1", "k", "v", ""],
        ["output2"],
        name="attention2",
    )

    graph = make_graph(
        [attention1, attention2],
        "graph",
        [
            make_tensor_value_info("q", TensorProto.FLOAT, [1, 16, 256]),
            make_tensor_value_info("k", TensorProto.FLOAT, [1, 16, 256]),
            make_tensor_value_info("v", TensorProto.FLOAT, [1, 16, 256]),
        ],
        [
            make_tensor_value_info("output1", TensorProto.FLOAT, [1, 16, 256]),
            make_tensor_value_info("output2", TensorProto.FLOAT, [1, 16, 256]),
        ],
        [
            from_array(np.random.randn(1, 16, 256).astype(np.float32), "k"),
            from_array(np.random.randn(1, 16, 256).astype(np.float32), "v"),
        ],
    )

    opset_imports = [make_operatorsetid("", opset)]
    return make_model(graph, opset_imports=opset_imports)


def test_one_attention_no_mask():
    """Test adding mask to a single attention layer without mask."""
    model = _make_model_with_one_attention(opset=23)
    graph = OnnxGraph(model)
    assert graph.opset_version == 23

    pm = PassManager(["attention_add_mask"])
    graph = pm.optimize(graph)

    attention = graph.nodes["attention"]["pb"]
    assert len(attention.input) >= 4
    assert attention.input[3] == "full_attention_mask"

    input_names = [inp.name for inp in graph.input]
    assert "full_attention_mask" in input_names

    mask_info = graph.tensor_info("full_attention_mask")
    assert mask_info[0] == [16, 1]
    assert mask_info[1] == TensorProto.INT32


def test_one_attention_already_has_mask():
    """Test that rewriter skips attention layer that already has a mask."""
    model = _make_model_with_one_attention(opset=23, has_mask=True)
    graph = OnnxGraph(model)
    assert graph.opset_version == 23

    original_input_count = len(graph.input)

    pm = PassManager(["attention_add_mask"])
    graph = pm.optimize(graph)

    attention = graph.nodes["attention"]["pb"]
    assert attention.input[3] == "attn_mask"
    assert len(graph.input) == original_input_count


def test_two_attention_no_mask():
    """Test adding mask to two attention layers without mask."""
    model = _make_model_with_two_attention(opset=23)
    graph = OnnxGraph(model)
    assert graph.opset_version == 23

    pm = PassManager(["attention_add_mask"])
    graph = pm.optimize(graph)

    attention1 = graph.nodes["attention1"]["pb"]
    attention2 = graph.nodes["attention2"]["pb"]

    assert len(attention1.input) >= 4
    assert attention1.input[3] == "full_attention_mask"
    assert len(attention2.input) >= 4
    assert attention2.input[3] == "full_attention_mask"

    input_names = [inp.name for inp in graph.input]
    assert "full_attention_mask" in input_names


def test_skips_low_opset():
    """Test that rewriter skips for opset < 23."""
    model = _make_model_with_one_attention(opset=18)
    graph = OnnxGraph(model)
    assert graph.opset_version == 18

    original_input_count = len(graph.input)

    pm = PassManager(["attention_add_mask"])
    graph = pm.optimize(graph)

    attention = graph.nodes["attention"]["pb"]
    if len(attention.input) > 3:
        assert attention.input[3] == ""
    assert len(graph.input) == original_input_count


def test_mask_already_in_graph_inputs():
    """Test reusing existing mask graph input."""
    model = _make_model_with_one_attention(
        opset=23, has_mask=False, mask_as_graph_input=True
    )
    graph = OnnxGraph(model)
    assert graph.opset_version == 23

    original_input_count = len(graph.input)

    pm = PassManager(
        ["attention_add_mask"],
        configs={"attention_add_mask": {"mask_name": "attn_mask"}},
    )
    graph = pm.optimize(graph)

    attention = graph.nodes["attention"]["pb"]
    assert attention.input[3] == "attn_mask"
    assert len(graph.input) == original_input_count


def test_custom_mask_name():
    """Test adding mask with custom name via parameter."""
    from onnxifier.passes.swap.attention_add_mask import AttentionAddMaskRewriter

    model = _make_model_with_one_attention(opset=23)
    graph = OnnxGraph(model)

    rewriter = AttentionAddMaskRewriter(mask_name="custom_mask")
    graph = rewriter(graph)

    attention = graph.nodes["attention"]["pb"]
    assert attention.input[3] == "custom_mask"

    input_names = [inp.name for inp in graph.input]
    assert "custom_mask" in input_names


def test_total_sequence_length_from_output_kv():
    """Test total_sequence_length inferred from present_key/present_value output."""
    attention = make_node(
        "Attention",
        ["q", "k", "v", ""],
        ["output", "present_key", "present_value"],
        name="attention",
    )
    graph = make_graph(
        [attention],
        "graph",
        [
            make_tensor_value_info("q", TensorProto.FLOAT, [1, 16, 256]),
            make_tensor_value_info("k", TensorProto.FLOAT, [1, 16, 256]),
            make_tensor_value_info("v", TensorProto.FLOAT, [1, 16, 256]),
        ],
        [
            make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 256]),
            make_tensor_value_info("present_key", TensorProto.FLOAT, [1, 16, 128, 256]),
            make_tensor_value_info(
                "present_value", TensorProto.FLOAT, [1, 16, 128, 256]
            ),
        ],
        [
            from_array(np.random.randn(1, 16, 256).astype(np.float32), "k"),
            from_array(np.random.randn(1, 16, 256).astype(np.float32), "v"),
        ],
    )
    model = make_model(graph, opset_imports=[make_operatorsetid("", 23)])
    graph = OnnxGraph(model)

    pm = PassManager(["attention_add_mask"])
    graph = pm.optimize(graph)

    mask_info = graph.tensor_info("full_attention_mask")
    assert mask_info[0] == [16, 128]
