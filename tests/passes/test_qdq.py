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
import onnx
from onnx import numpy_helper
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _build_graph1(bias=True):
    qconv = make_node(
        "QLinearConv",
        inputs=[
            "x",
            "x_scale",
            "x_zero_point",
            "w",
            "w_scale",
            "w_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        outputs=["y"],
        name="qconv",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    if bias:
        qconv.input.append("b")
    graph = make_graph(
        [qconv],
        "test_qconv",
        inputs=[make_tensor_value_info("x", onnx.TensorProto.UINT8, [1, 3, 16, 16])],
        outputs=[make_tensor_value_info("y", onnx.TensorProto.UINT8, [1, 16, 16, 16])],
        initializer=[
            numpy_helper.from_array(np.array(1 / 255, np.float32), "x_scale"),
            numpy_helper.from_array(np.array(0, np.uint8), "x_zero_point"),
            # avoid overflow
            numpy_helper.from_array(
                np.random.randint(0, 255, size=[16, 3, 3, 3], dtype=np.uint8), "w"
            ),
            numpy_helper.from_array(np.array([1 / 2048] * 16, np.float32), "w_scale"),
            numpy_helper.from_array(np.zeros([16], np.uint8), "w_zero_point"),
            numpy_helper.from_array(np.array(1 / 255, np.float32), "y_scale"),
            numpy_helper.from_array(np.array(0, np.uint8), "y_zero_point"),
            # this may occur some overflow
            numpy_helper.from_array(
                np.random.randint(-128 * 255, 127 * 255, size=[16], dtype=np.int32), "b"
            ),
        ],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def _build_graph2():
    m, n, k = 4, 8, 32
    qmatmul = make_node(
        "QLinearMatMul",
        inputs=[
            "a",
            "a_scale",
            "a_zero_point",
            "b",
            "b_scale",
            "b_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        outputs=["y"],
        name="qmatmul",
    )
    graph = make_graph(
        [qmatmul],
        "test_qmatmul",
        inputs=[
            make_tensor_value_info("a", onnx.TensorProto.UINT8, [m, n]),
            make_tensor_value_info("b", onnx.TensorProto.UINT8, [n, k]),
        ],
        outputs=[make_tensor_value_info("y", onnx.TensorProto.UINT8, [m, k])],
        initializer=[
            numpy_helper.from_array(np.array(1 / 255, np.float32), "a_scale"),
            numpy_helper.from_array(np.array(0, np.uint8), "a_zero_point"),
            numpy_helper.from_array(np.array(1 / 255, np.float32), "b_scale"),
            numpy_helper.from_array(np.array(0, np.uint8), "b_zero_point"),
            numpy_helper.from_array(np.array(1 / 255, np.float32), "y_scale"),
            numpy_helper.from_array(np.array(0, np.uint8), "y_zero_point"),
        ],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_qconv_to_qdq_conv():
    graph = _build_graph1()
    run1 = Evaluator(graph.model, "OnnxRuntime")
    pm = PassManager(["unfuse_qconv"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) > 1
    run2 = Evaluator(graph.model, "OnnxRuntime")

    x = np.random.randint(0, 255, size=[1, 3, 16, 16], dtype=np.uint8)
    y1 = run1(["y"], {"x": x})[0]
    y2 = run2(["y"], {"x": x})[0]
    err = np.abs(y1.astype(np.int32) - y2.astype(np.int32))
    assert err.max() <= 1

    pm = PassManager(["fuse_qconv"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 1
    run3 = Evaluator(graph.model, "OnnxRuntime")
    y3 = run3(["y"], {"x": x})[0]
    err = np.abs(y1.astype(np.int32) - y3.astype(np.int32))
    assert err.max() <= 1
    assert not (np.all(y1 == 0) or np.all(y1 == 255))


def test_qconv_to_qdq_conv_without_bias():
    graph = _build_graph1(bias=False)
    run1 = Evaluator(graph.model, "OnnxRuntime")
    pm = PassManager(["unfuse_qconv"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) > 1
    run2 = Evaluator(graph.model, "OnnxRuntime")

    x = np.random.randint(0, 255, size=[1, 3, 16, 16], dtype=np.uint8)
    y1 = run1(["y"], {"x": x})[0]
    y2 = run2(["y"], {"x": x})[0]
    err = np.abs(y1.astype(np.int32) - y2.astype(np.int32))
    assert err.max() <= 1

    pm = PassManager(["fuse_qconv"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 1
    run3 = Evaluator(graph.model, "OnnxRuntime")
    y3 = run3(["y"], {"x": x})[0]
    err = np.abs(y1.astype(np.int32) - y3.astype(np.int32))
    assert err.max() <= 1
    assert not (np.all(y1 == 0) or np.all(y1 == 255))


def test_qmatmul_to_qdq_matmul():
    graph = _build_graph2()
    run1 = Evaluator(graph.model, "OnnxRuntime")
    pm = PassManager(["unfuse_qmatmul"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) > 1
    run2 = Evaluator(graph.model, "OnnxRuntime")

    a = np.random.randint(0, 255, size=[4, 8], dtype=np.uint8)
    b = np.random.randint(0, 255, size=[8, 32], dtype=np.uint8)
    y1 = run1(["y"], {"a": a, "b": b})[0]
    y2 = run2(["y"], {"a": a, "b": b})[0]
    err = np.abs(y1.astype(np.int32) - y2.astype(np.int32))
    assert err.max() <= 1
