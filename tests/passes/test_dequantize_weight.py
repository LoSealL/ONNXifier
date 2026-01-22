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
from onnx.numpy_helper import from_array
from onnx.onnx_pb import TensorProto

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET
from onnxifier.checker import show_difference
from onnxifier.evaluator import Evaluator
from onnxifier.graph import OnnxGraph
from onnxifier.pass_manager import PassManager


def _make_conv_graph():
    cast = make_node(
        "Cast",
        inputs=["W"],
        outputs=["WF16"],
        to=TensorProto.FLOAT,
        name="cast",
    )
    mul = make_node(
        "Mul",
        inputs=["WF16", "scale"],
        outputs=["W_dequant"],
        name="mul",
    )
    dq = make_node(
        "DequantizeLinear",
        inputs=["X", "x_scale", "x_zp"],
        outputs=["X_dequant"],
        name="dq",
    )
    conv = make_node(
        "Conv",
        inputs=["X_dequant", "W_dequant"],
        outputs=["Y"],
        name="conv",
    )
    graph = make_graph(
        [dq, cast, mul, conv],
        name="test",
        inputs=[make_tensor_value_info("X", TensorProto.UINT8, [1, 8, 32, 32])],
        outputs=[make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, "H", "W"])],
        initializer=[
            from_array(np.random.uniform(-128, 127, [4, 8, 3, 3]).astype(np.int8), "W"),
            from_array(np.ones([4, 1, 1, 1], np.float32) / 127, "scale"),
            from_array(np.ones([8], np.float32) / 255, "x_scale"),
            from_array(np.zeros([8], np.uint8), "x_zp"),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def _make_matmul_graph():
    cast = make_node(
        "Cast",
        inputs=["W"],
        outputs=["WF16"],
        to=TensorProto.FLOAT,
        name="cast",
    )
    mul = make_node(
        "Mul",
        inputs=["WF16", "scale"],
        outputs=["W_dequant"],
        name="mul",
    )
    gemm = make_node(
        "MatMul",
        inputs=["X", "W_dequant"],
        outputs=["Y"],
        name="gemm",
    )
    graph = make_graph(
        [cast, mul, gemm],
        name="test",
        inputs=[make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, 16, 32])],
        outputs=[make_tensor_value_info("Y", TensorProto.FLOAT, [1, "C", "H", "W"])],
        initializer=[
            from_array(np.random.uniform(-128, 127, [8, 32, 16]).astype(np.int8), "W"),
            from_array(np.ones([8, 1, 1], np.float32), "scale"),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def _make_gemm_graph():
    cast = make_node(
        "Cast",
        inputs=["W"],
        outputs=["WF16"],
        to=TensorProto.FLOAT,
        name="cast",
    )
    mul = make_node(
        "Mul",
        inputs=["WF16", "scale"],
        outputs=["W_dequant"],
        name="mul",
    )
    gemm = make_node(
        "Gemm",
        inputs=["X", "W_dequant"],
        outputs=["Y"],
        name="gemm",
    )
    graph = make_graph(
        [cast, mul, gemm],
        name="test",
        inputs=[make_tensor_value_info("X", TensorProto.FLOAT, [8, 16])],
        outputs=[make_tensor_value_info("Y", TensorProto.FLOAT, ["M", "N"])],
        initializer=[
            from_array(np.random.uniform(-128, 127, [16, 8]).astype(np.int8), "W"),
            from_array(np.ones([16, 1], np.float32), "scale"),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_conv_dequantize_weight():
    graph = _make_conv_graph()
    runner1 = Evaluator(graph.model)
    pm = PassManager(["conv_dequantize_weight"])
    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model)

    x = np.random.randint(0, 255, size=[1, 8, 32, 32], dtype=np.uint8)
    y1 = runner1(["Y"], {"X": x})[0]
    y2 = runner2(["Y"], {"X": x})[0]

    assert np.allclose(y1, y2, rtol=1e-2), show_difference(y1, y2, rtol=1e-2)


def test_matmul_dequantize_weight():
    graph = _make_matmul_graph()
    runner1 = Evaluator(graph.model)
    pm = PassManager(["gemm_dequantize_weight"])
    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model)

    x = np.random.rand(1, 8, 16, 32).astype(np.float32)
    y1 = runner1(["Y"], {"X": x})[0]
    y2 = runner2(["Y"], {"X": x})[0]

    assert np.allclose(y1, y2, rtol=1e-6), show_difference(y1, y2, rtol=1e-6)


def test_gemm_dequantize_weight():
    graph = _make_gemm_graph()
    runner1 = Evaluator(graph.model)
    pm = PassManager(["gemm_dequantize_weight"])
    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model)

    x = np.random.rand(8, 16).astype(np.float32)
    y1 = runner1(["Y"], {"X": x})[0]
    y2 = runner2(["Y"], {"X": x})[0]

    assert np.allclose(y1, y2, rtol=1e-6), show_difference(y1, y2, rtol=1e-6)


def test_recalculate_dequantize_weight():
    graph = _make_conv_graph()
    runner1 = Evaluator(graph.model)
    pm = PassManager(["recalculate_dequantize_weight"])
    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model)

    x = np.random.rand(1, 8, 32, 32).astype(np.uint8)
    y1 = runner1(["Y"], {"X": x})[0]
    y2 = runner2(["Y"], {"X": x})[0]

    assert np.allclose(y1, y2, rtol=1e-6), show_difference(y1, y2, rtol=1e-6)
