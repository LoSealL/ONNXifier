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
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _build_graph():
    dq0 = make_node(
        "DequantizeLinear", ["qw0", "s0", "zp0"], ["w0"], axis=0, name="dq_w0"
    )
    conv = make_node("Conv", ["X", "w0"], ["a0"], auto_pad="SAME_UPPER", name="conv")
    act0 = make_node("Relu", ["a0"], ["b0"], name="act0")
    dq1 = make_node(
        "DequantizeLinear", ["qw1", "s1", "zp1"], ["w1"], axis=0, name="dq_w1"
    )
    matmul = make_node("MatMul", ["b0", "w1"], ["a1"], name="matmul")
    act1 = make_node("Relu", ["a1"], ["b1"], name="act1")
    dq2 = make_node(
        "DequantizeLinear", ["qw2", "s2", "zp2"], ["w2"], axis=1, name="dq_w2"
    )
    deconv = make_node(
        "ConvTranspose", ["b1", "w2"], ["a2"], auto_pad="SAME_UPPER", name="deconv"
    )
    act2 = make_node("Relu", ["a2"], ["Y"], name="act2")
    graph = make_graph(
        [dq0, conv, act0, dq1, matmul, act1, dq2, deconv, act2],
        "test_graph",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 224, 224])],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3, 224, 224])],
        initializer=[
            from_array(np.random.randint(0, 256, (32, 3, 3, 3), dtype=np.uint8), "qw0"),
            from_array(np.random.randint(0, 256, (224, 224), dtype=np.uint8), "qw1"),
            from_array(np.random.randint(0, 256, (32, 3, 3, 3), dtype=np.uint8), "qw2"),
            from_array(np.ones([32]).astype(np.float32), "s0"),
            from_array(np.ones([224]).astype(np.float32), "s1"),
            from_array(np.ones([3]).astype(np.float32), "s2"),
            from_array(np.zeros([32], dtype=np.uint8), "zp0"),
            from_array(np.zeros([224], dtype=np.uint8), "zp1"),
            from_array(np.zeros([3], dtype=np.uint8), "zp2"),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_canonicalize_uint8_weights():
    graph = _build_graph()
    runner1 = Evaluator(graph.model, "onnxruntime")
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    y1 = runner1(["Y"], {"X": x})[0]

    pm = PassManager(["canonicalize_uint8_weights"])
    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model, "onnxruntime")

    y2 = runner2(["Y"], {"X": x})[0]
    np.testing.assert_allclose(y1, y2, rtol=1e-2, atol=1e-2)
