"""
Copyright (C) 2024 The ONNXIFIER Authors.

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
import onnx
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _make_graph():
    ide = make_node("Identity", ["x"], ["id"])
    sigm = make_node("Sigmoid", ["id"], ["sigm"])
    mul = make_node("Mul", ["id", "sigm"], ["mul"])
    graph = make_graph(
        [ide, sigm, mul],
        "test",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, (1, 16, 24, 24))],
        [make_tensor_value_info("mul", onnx.TensorProto.FLOAT, (1, 16, 24, 24))],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def _make_graph_inline():
    ide1 = make_node("Identity", ["x"], ["id1"], "1")
    sigm1 = make_node("Sigmoid", ["id1"], ["sigm1"], "2")
    mul1 = make_node("Mul", ["id1", "sigm1"], ["mul1"], "3")
    ide2 = make_node("Identity", ["mul1"], ["id2"], "4")
    sigm2 = make_node("Sigmoid", ["id2"], ["sigm2"], "5")
    mul2 = make_node("Mul", ["id2", "sigm2"], ["mul2"], "6")
    graph = make_graph(
        [ide1, sigm1, mul1, ide2, sigm2, mul2],
        "test",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, (1, 16, 24, 24))],
        [make_tensor_value_info("mul2", onnx.TensorProto.FLOAT, (1, 16, 24, 24))],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_fuse_swish():
    graph = _make_graph()
    runner1 = Evaluator(graph.model, "OnnxRuntime")

    pm = PassManager(["fuse_swish"])
    graph = pm.optimize(graph, strict=True)

    assert len(graph) == 2
    for node in graph:
        if graph.nodes[node]["pb"].op_type != "Identity":
            assert graph.nodes[node]["pb"].op_type == "Swish"

    # only onnxruntime can infer functions
    runner2 = Evaluator(graph.model, "OnnxRuntime")

    x = np.random.randn(1, 16, 24, 24).astype(np.float32)
    y1 = runner1([], {"x": x})[0]
    y2 = runner2([], {"x": x})[0]
    assert np.allclose(y1, y2)


def test_fuse_inline_swish():
    graph = _make_graph_inline()
    assert set(graph) == {str(i) for i in range(1, 7)}
    runner1 = Evaluator(graph.model, "OnnxRuntime")

    pm = PassManager(["fuse_swish"])
    graph = pm.optimize(graph, strict=True)

    assert len(graph) == 4
    for node in graph:
        if graph.nodes[node]["pb"].op_type != "Identity":
            assert graph.nodes[node]["pb"].op_type == "Swish"

    pm = PassManager(["inline_functions"])
    graph = pm.optimize(graph, strict=True)
    assert set(graph) == {str(i) for i in range(1, 7)}
    for name in graph:
        node = graph.nodes[name]["pb"]
        if name in {"2", "5"}:
            assert node.op_type == "Sigmoid"
        elif name in {"3", "6"}:
            assert node.op_type == "Mul"

    # only onnxruntime can infer functions
    runner2 = Evaluator(graph.model, "OnnxRuntime")

    x = np.random.randn(1, 16, 24, 24).astype(np.float32)
    y1 = runner1([], {"x": x})[0]
    y2 = runner2([], {"x": x})[0]
    assert np.allclose(y1, y2)
