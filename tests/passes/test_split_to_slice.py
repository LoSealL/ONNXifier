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

# pylint: disable=missing-docstring

import numpy as np
import onnx
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _make_graph1():
    split = make_node("Split", ["x", "s"], ["s0", "s1"], name="split", axis=0)
    relu1 = make_node("Relu", ["s0"], ["y0"], name="relu1")
    relu2 = make_node("Relu", ["s1"], ["y1"], name="relu2")
    graph = make_graph(
        [split, relu1, relu2],
        "test",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, [32])],
        [
            make_tensor_value_info("y0", onnx.TensorProto.FLOAT, [24]),
            make_tensor_value_info("y1", onnx.TensorProto.FLOAT, [8]),
        ],
        [onnx.numpy_helper.from_array(np.array([24, 8], dtype=np.int64), "s")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def _make_graph2():
    split = make_node(
        "Split", ["x"], ["s0", "s1", "s2"], name="split", axis=0, num_outputs=3
    )
    relu1 = make_node("Relu", ["s0"], ["y0"], name="relu1")
    relu2 = make_node("Relu", ["s1"], ["y1"], name="relu2")
    relu3 = make_node("Relu", ["s2"], ["y2"], name="relu3")
    graph = make_graph(
        [split, relu1, relu2, relu3],
        "test",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, [8])],
        [
            make_tensor_value_info("y0", onnx.TensorProto.FLOAT, ["1"]),
            make_tensor_value_info("y1", onnx.TensorProto.FLOAT, ["2"]),
            make_tensor_value_info("y2", onnx.TensorProto.FLOAT, ["3"]),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_split_to_slice():
    graph = _make_graph1()
    run1 = Evaluator(graph.model)
    x = np.random.randn(32).astype(np.float32)
    y01, y11 = run1(["y0", "y1"], {"x": x})
    assert y01.shape == (24,)
    assert y11.shape == (8,)

    pm = PassManager(["split_to_slice"])
    graph = pm.optimize(graph, strict=True)
    run2 = Evaluator(graph.model)
    y02, y12 = run2(["y0", "y1"], {"x": x})
    error0 = np.abs(y01 - y02).max()
    error1 = np.abs(y11 - y12).max()
    assert np.allclose(y01, y02), f"error0={error0}"
    assert np.allclose(y11, y12), f"error1={error1}"


def test_split_to_slice_opset18():
    graph = _make_graph2()
    run1 = Evaluator(graph.model)
    x = np.random.randn(8).astype(np.float32)
    y01, y11, y21 = run1(["y0", "y1", "y2"], {"x": x})
    assert y01.shape == (3,)
    assert y11.shape == (3,)
    assert y21.shape == (2,)

    pm = PassManager(["split_to_slice"])
    graph = pm.optimize(graph, strict=True)
    run2 = Evaluator(graph.model)
    y02, y12, y22 = run2(["y0", "y1", "y2"], {"x": x})
    error0 = np.abs(y01 - y02).max()
    error1 = np.abs(y11 - y12).max()
    error2 = np.abs(y21 - y22).max()
    assert np.allclose(y01, y02), f"error0={error0}"
    assert np.allclose(y11, y12), f"error1={error1}"
    assert np.allclose(y21, y22), f"error2={error2}"
