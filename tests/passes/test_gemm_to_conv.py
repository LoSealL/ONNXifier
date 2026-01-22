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
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_type_proto,
    make_value_info,
)
from onnx.reference import ReferenceEvaluator

from onnxifier import PassManager
from onnxifier.graph import OnnxGraph


def _build_test_graph(alpha, beta, transA, transB):
    gemm = make_node(
        "Gemm",
        inputs=["x", "B", "C"],
        outputs=["y"],
        name="gemm",
        alpha=float(alpha),
        beta=float(beta),
        transA=transA,
        transB=transB,
    )
    x_shape = (64, 1) if transA else (1, 64)
    b_shape = (32, 64) if transB else (64, 32)
    graph = make_graph(
        [gemm],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, x_shape))],
        [make_value_info("y", make_tensor_type_proto(1, [1, 32]))],
        [
            make_tensor("B", 1, b_shape, np.random.rand(*b_shape)),
            make_tensor("C", 1, [32], np.random.rand(32)),
        ],
    )
    return make_model(graph)


def _build_quantize_test_graph(alpha, beta, transA, transB):
    gemm = make_node(
        "Gemm",
        inputs=["A", "B", "C"],
        outputs=["Y"],
        name="gemm",
        alpha=float(alpha),
        beta=float(beta),
        transA=transA,
        transB=transB,
    )
    dequant = make_node(
        "DequantizeLinear",
        inputs=["x", "x_scale", "x_zero_point"],
        outputs=["B"],
        name="dequantzie",
        axis=0 if transB else 1,
    )

    a_shape = (64, 1) if transA else (1, 64)
    b_shape = (32, 64) if transB else (64, 32)
    graph = make_graph(
        [dequant, gemm],
        "graph",
        [make_value_info("A", make_tensor_type_proto(1, a_shape))],
        [make_value_info("Y", make_tensor_type_proto(1, [1, 32]))],
        [
            make_tensor("C", 1, [32], np.random.rand(32)),
            make_tensor(
                "x", 3, b_shape, np.random.randint(-128, 128, b_shape, dtype="int8")
            ),
            make_tensor("x_scale", 1, [32], np.random.rand(32)),
            make_tensor(
                "x_zero_point",
                3,
                [32],
                np.random.randint(-128, 128, [32], dtype="int8"),
            ),
        ],
    )
    return make_model(graph)


def _build_test_ND_graph(shape1, shape2):
    matmul = make_node(
        "MatMul",
        inputs=["x", "w"],
        outputs=["y"],
        name="matmul",
    )
    x_shape = shape1
    b_shape = shape2
    out_shape = np.maximum(shape1[:-2], shape2[:-2]).tolist() + [shape1[-2], shape2[-1]]
    graph = make_graph(
        [matmul],
        "graph",
        [
            make_value_info("x", make_tensor_type_proto(1, x_shape)),
            make_value_info("w", make_tensor_type_proto(1, b_shape)),
        ],
        [make_value_info("y", make_tensor_type_proto(1, out_shape))],
        [],
    )
    return make_model(graph)


def test_rewriter():
    graph = OnnxGraph(_build_test_graph(1, 1, 0, 0))
    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)


def test_rewriter_transB():
    graph = OnnxGraph(_build_test_graph(1, 1, 0, 1))
    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)


def test_rewriter_transA():
    graph = OnnxGraph(_build_test_graph(1, 1, 1, 0))
    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)


def test_quantize_rewriter():
    graph = OnnxGraph(_build_quantize_test_graph(1, 1, 0, 0))
    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)


def test_quantize_rewriter_transA():
    graph = OnnxGraph(_build_quantize_test_graph(1, 1, 1, 0))
    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)


def test_quantize_rewriter_transB():
    graph = OnnxGraph(_build_quantize_test_graph(1, 1, 0, 1))
    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)


def test_quantize_rewriter_scale_alpha():
    graph = OnnxGraph(_build_quantize_test_graph(0.5, 1, 0, 1))
    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)


def test_rewriter_3D():
    x_shape = [128, 4, 64]
    w_shape = [1, 64, 16]
    graph = OnnxGraph(_build_test_ND_graph(x_shape, w_shape))
    runner1 = ReferenceEvaluator(graph.model)
    x = np.random.rand(*x_shape).astype("float32")
    w = np.random.rand(*w_shape).astype("float32")
    y1 = runner1.run(None, {"x": x, "w": w})

    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)

    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x": x, "w": w})
    assert np.allclose(y1, y2)


def test_rewriter_3_2D():
    x_shape = [1, 4, 64]
    w_shape = [64, 16]
    graph = OnnxGraph(_build_test_ND_graph(x_shape, w_shape))
    runner1 = ReferenceEvaluator(graph.model)
    x = np.random.rand(*x_shape).astype("float32")
    w = np.random.rand(*w_shape).astype("float32")
    y1 = runner1.run(None, {"x": x, "w": w})

    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)

    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x": x, "w": w})
    assert np.allclose(y1, y2)


def test_rewriter_3ND_swap():
    x_shape = [1, 4, 64]
    w_shape = [128, 64, 16]
    graph = OnnxGraph(_build_test_ND_graph(x_shape, w_shape))
    runner1 = ReferenceEvaluator(graph.model)
    x = np.random.rand(*x_shape).astype("float32")
    w = np.random.rand(*w_shape).astype("float32")
    y1 = runner1.run(None, {"x": x, "w": w})

    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)

    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x": x, "w": w})
    assert np.allclose(y1, y2)


def test_rewriter_4D():
    x_shape = [1, 128, 4, 64]
    w_shape = [1, 1, 64, 16]
    graph = OnnxGraph(_build_test_ND_graph(x_shape, w_shape))
    runner1 = ReferenceEvaluator(graph.model)
    x = np.random.rand(*x_shape).astype("float32")
    w = np.random.rand(*w_shape).astype("float32")
    y1 = runner1.run(None, {"x": x, "w": w})

    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)

    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x": x, "w": w})
    assert np.allclose(y1, y2)


def test_rewriter_4ND_swap():
    x_shape = [1, 1, 4, 64]
    w_shape = [1, 128, 64, 16]
    graph = OnnxGraph(_build_test_ND_graph(x_shape, w_shape))
    runner1 = ReferenceEvaluator(graph.model)
    x = np.random.rand(*x_shape).astype("float32")
    w = np.random.rand(*w_shape).astype("float32")
    y1 = runner1.run(None, {"x": x, "w": w})

    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)

    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x": x, "w": w})
    assert np.allclose(y1, y2)


def test_rewriter_5D():
    x_shape = [1, 1, 128, 4, 64]
    w_shape = [1, 1, 1, 64, 16]
    graph = OnnxGraph(_build_test_ND_graph(x_shape, w_shape))
    runner1 = ReferenceEvaluator(graph.model)
    x = np.random.rand(*x_shape).astype("float32")
    w = np.random.rand(*w_shape).astype("float32")
    y1 = runner1.run(None, {"x": x, "w": w})

    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)

    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x": x, "w": w})
    assert np.allclose(y1, y2)


def test_rewriter_5ND_swap():
    x_shape = [1, 1, 1, 4, 64]
    w_shape = [1, 2, 128, 64, 16]
    graph = OnnxGraph(_build_test_ND_graph(x_shape, w_shape))
    runner1 = ReferenceEvaluator(graph.model)
    x = np.random.rand(*x_shape).astype("float32")
    w = np.random.rand(*w_shape).astype("float32")
    y1 = runner1.run(None, {"x": x, "w": w})

    pm = PassManager(["initializer_to_constant", "gemm_to_conv"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)

    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x": x, "w": w})
    assert np.allclose(y1, y2)
