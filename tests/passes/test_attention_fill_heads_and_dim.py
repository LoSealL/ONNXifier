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

import math

import numpy as np
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier import OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _get_attrs(node):
    return {a.name: a for a in node.attribute}


def _make_attention_model(
    q_shape,
    k_shape,
    v_shape,
    *,
    attrs=None,
    inputs=None,
    opset=24,
):
    node = make_node(
        "Attention", inputs or ["q", "k", "v"], ["y"], name="attn", **(attrs or {})
    )
    graph = make_graph(
        [node],
        "g",
        [
            make_value_info("q", make_tensor_type_proto(1, q_shape)),
            make_value_info("k", make_tensor_type_proto(1, k_shape)),
            make_value_info("v", make_tensor_type_proto(1, v_shape)),
        ],
        [make_value_info("y", make_tensor_type_proto(1, None))],
    )
    return make_model(graph, opset_imports=[make_operatorsetid("", opset)])


def _assert_output_consistent(model, feeds):
    graph = OnnxGraph(model)
    runner_before = Evaluator(graph.model, backend="onnx")
    y_before = runner_before(["y"], feeds)[0]

    graph = PassManager(["attention_fill_heads_and_dim"]).optimize(graph, strict=True)
    runner_after = Evaluator(graph.model, backend="onnx")
    y_after = runner_after(["y"], feeds)[0]

    np.testing.assert_allclose(y_before, y_after, rtol=1e-5, atol=1e-5)
    return graph


def test_fill_heads_and_scale_from_4d_shapes():
    model = _make_attention_model(
        [2, 8, 16, 64],
        [2, 4, 32, 64],
        [2, 4, 32, 64],
        opset=24,
    )
    q = np.random.randn(2, 8, 16, 64).astype(np.float32)
    k = np.random.randn(2, 4, 32, 64).astype(np.float32)
    v = np.random.randn(2, 4, 32, 64).astype(np.float32)
    graph = _assert_output_consistent(model, {"q": q, "k": k, "v": v})
    node = graph.nodes["attn"]["pb"]
    attrs = _get_attrs(node)

    assert attrs["q_num_heads"].i == 8
    assert attrs["kv_num_heads"].i == 4
    assert math.isclose(attrs["scale"].f, 1.0 / math.sqrt(64.0), rel_tol=1e-6)

    assert attrs["is_causal"].i == 0
    assert attrs["qk_matmul_output_mode"].i == 0
    assert math.isclose(attrs["softcap"].f, 0.0, abs_tol=1e-12)


def test_fill_heads_and_scale_from_3d_shapes_with_heads_attrs():
    model = _make_attention_model(
        [2, 16, 64],
        [2, 16, 32],
        [2, 16, 32],
        attrs={"q_num_heads": 8, "kv_num_heads": 4},
        opset=23,
    )
    q = np.random.randn(2, 16, 64).astype(np.float32)
    k = np.random.randn(2, 16, 32).astype(np.float32)
    v = np.random.randn(2, 16, 32).astype(np.float32)
    graph = _assert_output_consistent(model, {"q": q, "k": k, "v": v})
    node = graph.nodes["attn"]["pb"]
    attrs = _get_attrs(node)

    assert attrs["kv_num_heads"].i == 4
    assert attrs["q_num_heads"].i == 8
    assert math.isclose(attrs["scale"].f, 1.0 / math.sqrt(8.0), rel_tol=1e-6)


def test_skip_uninferable_scale_but_fill_schema_defaults():
    model = _make_attention_model(
        [2, "q_seq", "q_hidden"],
        [2, "k_seq", "k_hidden"],
        [2, "k_seq", "k_hidden"],
        attrs={"q_num_heads": 4, "kv_num_heads": 4},
        opset=24,
    )
    q = np.random.randn(2, 8, 64).astype(np.float32)
    k = np.random.randn(2, 8, 64).astype(np.float32)
    v = np.random.randn(2, 8, 64).astype(np.float32)
    graph = _assert_output_consistent(model, {"q": q, "k": k, "v": v})
    node = graph.nodes["attn"]["pb"]
    attrs = _get_attrs(node)

    assert "scale" not in attrs
    assert attrs["q_num_heads"].i == 4
    assert attrs["kv_num_heads"].i == 4

    assert attrs["is_causal"].i == 0
    assert attrs["qk_matmul_output_mode"].i == 0
    assert math.isclose(attrs["softcap"].f, 0.0, abs_tol=1e-12)


def test_override_existing_inferable_attrs():
    model = _make_attention_model(
        [2, 4, 16, 32],
        [2, 4, 16, 32],
        [2, 4, 16, 32],
        attrs={"q_num_heads": 6, "scale": 0.125, "is_causal": 1},
        opset=24,
    )
    q = np.random.randn(2, 4, 16, 32).astype(np.float32)
    k = np.random.randn(2, 4, 16, 32).astype(np.float32)
    v = np.random.randn(2, 4, 16, 32).astype(np.float32)
    graph = _assert_output_consistent(model, {"q": q, "k": k, "v": v})
    node = graph.nodes["attn"]["pb"]
    attrs = _get_attrs(node)

    assert attrs["q_num_heads"].i == 4
    assert math.isclose(attrs["scale"].f, 0.125, rel_tol=1e-6)
    assert attrs["is_causal"].i == 1
    # Missing kv heads should still be inferred from 4D K shape.
    assert attrs["kv_num_heads"].i == 4
