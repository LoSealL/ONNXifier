"""
Copyright (C) 2025 The ONNXIFIER Authors.

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
import pytest
from onnx import numpy_helper
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.passes.canonicalization.reset_reshape_batch import trace_batch_dimension


def _make_graph():
    # a.T + reshape(b) + c.squeeze(1) + d.unsqueeze(-1) + gather(e)
    a = make_node("Transpose", ["a"], ["at"], perm=[0, 2, 1], name="at")
    b = make_node("Reshape", ["b", "bs"], ["br"], name="br")
    c = make_node("Squeeze", ["c", "ca"], ["cs"], name="cs")
    d = make_node("Unsqueeze", ["d", "da"], ["du"], name="du")
    e = make_node("Gather", ["e", "ei"], ["eg"], axis=3, name="eg")
    ca = make_node("Constant", [], ["ca"], value_int=1)
    add = make_node("Sum", ["at", "br", "cs", "du", "eg"], ["y"], name="add")
    graph = make_graph(
        [a, b, ca, c, d, e, add],
        "test",
        [
            make_tensor_value_info("a", onnx.TensorProto.FLOAT, [1, 128, 32]),
            make_tensor_value_info("b", onnx.TensorProto.FLOAT, [1, 32 * 128]),
            make_tensor_value_info("c", onnx.TensorProto.FLOAT, [1, 1, 1, 128]),
            make_tensor_value_info("d", onnx.TensorProto.FLOAT, [1, 1]),
            make_tensor_value_info("e", onnx.TensorProto.FLOAT, [1, 32, 1, 2]),
        ],
        [
            make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 32, 128]),
        ],
        [
            numpy_helper.from_array(np.array([1, 32, 128], np.int64), "bs"),
            # numpy_helper.from_array(np.array(1, np.int64), "ca"),
            numpy_helper.from_array(np.array([-1], np.int64), "da"),
            numpy_helper.from_array(np.array(0, np.int64), "ei"),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )

    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def _make_graph2():
    a = make_node("Transpose", ["a"], ["at"], perm=[1, 2, 0], name="at")
    b = make_node("Reshape", ["at", "bs"], ["br"], name="br")
    graph = make_graph(
        [a, b],
        "test",
        [make_tensor_value_info("a", onnx.TensorProto.FLOAT, [1, 32, 128])],
        [make_tensor_value_info("br", onnx.TensorProto.FLOAT, [32 * 128, 1])],
        [numpy_helper.from_array(np.array([32 * 128, 1], np.int64), "bs")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def _make_graph3():
    a = make_node("Transpose", ["a"], ["at"], perm=[1, 2, 0], name="at")
    b = make_node("Reshape", ["at", "bs"], ["br"], name="br")
    graph = make_graph(
        [a, b],
        "test",
        [make_tensor_value_info("a", onnx.TensorProto.FLOAT, [2, 128, 32])],
        [make_tensor_value_info("br", onnx.TensorProto.FLOAT, [128, 64])],
        [numpy_helper.from_array(np.array([128, 64], np.int64), "bs")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def _make_bad_graph():
    a = make_node("Transpose", ["a"], ["at"], perm=[1, 2, 0], name="at")
    b = make_node("Reshape", ["at", "bs"], ["br"], name="br")
    graph = make_graph(
        [a, b],
        "test",
        [make_tensor_value_info("a", onnx.TensorProto.FLOAT, [2, 32, 128])],
        [make_tensor_value_info("br", onnx.TensorProto.FLOAT, [64, 128])],
        [numpy_helper.from_array(np.array([64, 128], np.int64), "bs")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def _make_bad_graph2():
    a = make_node("Transpose", ["a"], ["at"], perm=[1, 2, 0], name="at")
    b = make_node("Reshape", ["at", "bs"], ["br"], name="br")
    graph = make_graph(
        [a, b],
        "test",
        [make_tensor_value_info("a", onnx.TensorProto.FLOAT, [2, 32, 128])],
        [make_tensor_value_info("br", onnx.TensorProto.FLOAT, [32, 2, 128])],
        [numpy_helper.from_array(np.array([32, 2, 128], np.int64), "bs")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def test_trace_batch_dimension():
    graph = _make_graph()
    tracing = trace_batch_dimension(graph)
    assert tracing["at"] == {0}
    assert tracing["br"] == {0}
    assert tracing["cs"] == {0}
    assert tracing["du"] == {0}
    assert tracing["eg"] == {0}
    assert tracing["add"] == {0}


def test_reset_reshape_batch():
    graph = _make_graph()
    pm = PassManager(["reset_reshape_batch"])
    graph = pm.optimize(graph, strict=True)

    reshape = graph.onnx_predecessors(graph.nodes["br"]["pb"])[-1].attribute[0].t
    assert -1 == numpy_helper.to_array(reshape)[0]


def test_reset_reshape_batch_to_last_dim():
    graph = _make_graph2()
    pm = PassManager(["reset_reshape_batch"])
    graph = pm.optimize(graph, strict=True)

    reshape = graph.onnx_predecessors(graph.nodes["br"]["pb"])[-1].attribute[0].t
    assert -1 == numpy_helper.to_array(reshape)[-1]


def test_reset_reshape_batch_merged_dim():
    graph = _make_graph3()
    pm = PassManager(["reset_reshape_batch"])
    graph = pm.optimize(graph, strict=True)

    reshape = graph.onnx_predecessors(graph.nodes["br"]["pb"])[-1].attribute[0].t
    assert -1 == numpy_helper.to_array(reshape)[-1]


def test_reset_reshape_batch_bad_graph():
    pm = PassManager(["reset_reshape_batch"])

    graph = _make_bad_graph()
    with pytest.raises((ValueError, NotImplementedError)):
        pm.optimize(graph, strict=True)

    graph = _make_bad_graph2()
    with pytest.raises((ValueError, NotImplementedError)):
        pm.optimize(graph, strict=True)
