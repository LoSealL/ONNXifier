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

import json

import numpy as np
import onnx
from onnx.helper import (
    make_function,
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _make_graph():
    f_log = make_node("Log", ["a0"], ["a1"])
    f_sm = make_node("Softmax", ["a1"], ["a2"], axis=1)
    f_sm.attribute[0].ref_attr_name = "dim"
    fn = make_function(
        "test",
        "MyLogSoftmax",
        ["a0"],
        ["a2"],
        [f_log, f_sm],
        opset_imports=[ONNXIFIER_OPSET],
        attributes=["dim"],
    )
    # note: use a different attribute value
    log_softmax = make_node("MyLogSoftmax", ["X"], ["Y"], dim=2, domain="test")
    graph = make_graph(
        [log_softmax],
        "test",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 2, 3])],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 2, 3])],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid("test", 1)],
        functions=[fn],
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def _make_graph_doc_string():
    f_log = make_node("Log", ["a0"], ["a1"])
    f_sm = make_node("Softmax", ["a1"], ["a2"], axis=1)
    f_sm.attribute[0].ref_attr_name = "dim"
    fn = make_function(
        "test",
        "MyLogSoftmax",
        ["a0"],
        ["a2"],
        [f_log, f_sm],
        opset_imports=[ONNXIFIER_OPSET],
        attributes=["dim"],
    )
    # note: use a different attribute value
    log_softmax = make_node(
        "MyLogSoftmax",
        ["X"],
        ["Y"],
        dim=2,
        domain="test",
        doc_string=json.dumps(["a", "b"]),
    )
    graph = make_graph(
        [log_softmax],
        "test",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 2, 3])],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 2, 3])],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid("test", 1)],
        functions=[fn],
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_inline_functions():
    graph = _make_graph()
    runner1 = Evaluator(graph.model, "OnnxRuntime")

    pm = PassManager(
        ["inline_functions"], configs={"inline_functions": {"force": True}}
    )
    graph = pm.optimize(graph, strict=True)
    assert len(graph) == 2

    runner2 = Evaluator(graph.model, "OnnxRuntime")
    x = np.random.uniform(1, 10, size=[1, 2, 3]).astype(np.float32)
    y1 = runner1(["Y"], {"X": x})[0]
    y2 = runner2(["Y"], {"X": x})[0]
    assert np.allclose(y1, y2)


def test_inline_functions_doc_string():
    graph = _make_graph_doc_string()
    runner1 = Evaluator(graph.model, "OnnxRuntime")

    pm = PassManager(
        ["inline_functions"], configs={"inline_functions": {"force": True}}
    )
    graph = pm.optimize(graph, strict=True)
    assert set(graph) == {"a", "b"}

    runner2 = Evaluator(graph.model, "OnnxRuntime")
    x = np.random.uniform(1, 10, size=[1, 2, 3]).astype(np.float32)
    y1 = runner1(["Y"], {"X": x})[0]
    y2 = runner2(["Y"], {"X": x})[0]
    assert np.allclose(y1, y2)
