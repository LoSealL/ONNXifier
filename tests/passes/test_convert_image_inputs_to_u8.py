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
import pytest
from onnx import TensorProto, checker
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.reference import ReferenceEvaluator

from onnxifier import ONNXIFIER_OPSET, PassManager
from onnxifier.graph import OnnxGraph


def _build_model():
    relu = make_node("Relu", ["x"], ["y"], "relu")
    graph = make_graph(
        [relu],
        "graph",
        [make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])],
        [],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def _optimize(configs=None):
    graph = OnnxGraph(_build_model())
    pm = PassManager(["convert_image_inputs_to_u8"], configs=configs)
    return pm.optimize(graph, strict=True)


def test_cast_only():
    graph = _optimize()
    shape, dtype = graph.tensor_info("x")
    assert dtype == TensorProto.UINT8
    assert shape == [1, 3, 8, 8]
    assert "x/cast" in graph
    assert graph.nodes["x/cast"]["pb"].op_type == "Cast"
    assert graph.nodes["relu"]["pb"].input[0] == "x/cast_output"
    checker.check_model(graph.model)

    rng = np.random.default_rng(42)
    x = rng.integers(0, 256, (1, 3, 8, 8)).astype(np.uint8)
    expected = np.maximum(x.astype(np.float32), 0)
    got = ReferenceEvaluator(graph.model).run(None, {"x": x})[0]
    np.testing.assert_allclose(got, expected)


def test_packed_mean_std():
    mean = [104.0, 117.0, 123.0]
    std = [58.0, 57.0, 57.0]
    graph = _optimize(
        {"convert_image_inputs_to_u8": {"packed": True, "mean": mean, "std": std}}
    )
    shape, dtype = graph.tensor_info("x")
    assert dtype == TensorProto.UINT8
    assert shape == [1, 8, 8, 3]
    assert "x/transpose" in graph
    assert "x/cast" in graph
    assert "x/add" in graph
    assert "x/mul" in graph
    assert graph.nodes["relu"]["pb"].input[0] == "x/mul_output"
    checker.check_model(graph.model)

    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (1, 8, 8, 3)).astype(np.uint8)
    ref = x.astype(np.float32).transpose(0, 3, 1, 2)
    ref = (ref - np.asarray(mean).reshape(1, 3, 1, 1)) / np.asarray(std).reshape(
        1, 3, 1, 1
    )
    expected = np.maximum(ref, 0)
    got = ReferenceEvaluator(graph.model).run(None, {"x": x})[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_explicit_input_names():
    graph = _optimize({"convert_image_inputs_to_u8": {"input_names": ["x"]}})
    assert graph.tensor_type("x") == TensorProto.UINT8
    assert "x/cast" in graph


def test_unknown_input_name():
    with pytest.raises(ValueError, match="not a graph input"):
        _optimize({"convert_image_inputs_to_u8": {"input_names": ["nope"]}})


def test_mean_std_auto_expand():
    mean, std = 10.0, 2.0
    graph = _optimize({"convert_image_inputs_to_u8": {"mean": mean, "std": std}})
    assert "x/add" in graph
    assert "x/mul" in graph
    checker.check_model(graph.model)

    rng = np.random.default_rng(1)
    x = rng.integers(0, 256, (1, 3, 8, 8)).astype(np.uint8)
    expected = np.maximum((x.astype(np.float32) - mean) / std, 0)
    got = ReferenceEvaluator(graph.model).run(None, {"x": x})[0]
    np.testing.assert_allclose(got, expected)


def test_mean_length_mismatch():
    with pytest.raises(ValueError, match="scalar"):
        _optimize({"convert_image_inputs_to_u8": {"mean": [1.0, 2.0]}})


def test_no_image_input():
    relu = make_node("Relu", ["z"], ["y"], "relu")
    graph = make_graph(
        [relu],
        "graph",
        [make_tensor_value_info("z", TensorProto.FLOAT, [8, 8])],
        [make_tensor_value_info("y", TensorProto.FLOAT, [8, 8])],
        [],
    )
    g = OnnxGraph(make_model(graph, opset_imports=[ONNXIFIER_OPSET]))
    pm = PassManager(["convert_image_inputs_to_u8"])
    out = pm.optimize(g, strict=True)
    assert len(out.nodes) == 1
