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

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _make_weights(make_zero):
    def _dice():
        return np.random.randint(1, 7)

    weights = {
        "W0": np.random.randn(32, 16, 3, 3).astype(np.float32),
        "B0": np.random.randn(32).astype(np.float32),
        "W1": np.random.randn(32, 32, 3, 3).astype(np.float32),
        "B1": np.random.randn(32).astype(np.float32),
        "W2": np.random.randn(16, 64, 3, 3).astype(np.float32),
        "B2": np.random.randn(16).astype(np.float32),
        "M2": np.random.randn(32, 32, 2, 2).astype(np.float32),
        "S0": np.random.randn(1, 32, 1, 1).astype(np.float32),
    }
    if make_zero:
        weights["W0"][[0, 1, 2, 3]] = 0
        weights["B0"][[0, 1, 2, 3]] = 0
        weights["W1"][:, [4, 5, 6, 7]] = 0
        weights["W1"][[1, 3, 5, 7]] = 0
        weights["W2"][:, [2, 4, 6, 8, 10]] = 0
        weights["M2"][:, [-1, -2, -3]] = 0
        weights["S0"][:, [1, 3, 5, 7, 9]] = 0
    return [from_array(v, k) for k, v in weights.items()]


def _make_graph(make_zero, has_add=False):
    r"""Conv
         |  \
         |   \
       (Add)  MaxPool
         |       |
         |      Conv
         |       |
         \   ConvTranspose
          \     /
          Concat
             |
           Conv
    """

    conv0 = make_node(
        "Conv", ["X", "W0", "B0"], ["C0"], auto_pad="SAME_UPPER", name="conv0"
    )
    act0 = make_node("Relu", ["C0"], ["A0"], name="act0")
    pool0 = make_node("MaxPool", ["A0"], ["P0"], kernel_shape=[2, 2], strides=[2, 2])
    conv1 = make_node(
        "Conv", ["P0", "W1", "B1"], ["C1"], auto_pad="SAME_UPPER", name="conv1"
    )
    act1 = make_node("Relu", ["C1"], ["A1"], name="act1")
    deconv = make_node(
        "ConvTranspose",
        ["A1", "M2"],
        ["D0"],
        auto_pad="SAME_UPPER",
        kernel_shape=[2, 2],
        strides=[2, 2],
        name="deconv",
    )
    add = make_node("Add", ["C0", "S0"], ["A2"], name="add")
    cat = make_node(
        "Concat", ["A2" if has_add else "C0", "D0"], ["Y"], axis=1, name="cat"
    )
    conv2 = make_node(
        "Conv", ["Y", "W2", "B2"], ["Z"], auto_pad="SAME_UPPER", name="conv2"
    )
    nodes = [conv0, act0, pool0, conv1, act1, deconv, cat, conv2]
    if has_add:
        nodes.insert(5, add)
    graph = make_graph(
        nodes,
        "test_graph",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 16, 32, 32])],
        [make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [1, 16, 32, 32])],
        _make_weights(make_zero),
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def test_remove_zero_channels_on_non_pruned_graph():
    graph = _make_graph(False)

    runner1 = Evaluator(graph.model, backend="OnnxRuntime")
    pm = PassManager(["remove_zero_channels", "onnx_optimizer"])
    graph = pm.optimize(graph, strict=True)

    runner2 = Evaluator(graph.model, backend="OnnxRuntime")

    x = np.random.randn(1, 16, 32, 32).astype(np.float32)
    y0 = runner1(["Z"], {"X": x})[0]
    y1 = runner2(["Z"], {"X": x})[0]

    np.testing.assert_allclose(y0, y1)


def test_remove_zero_channels_on_pruned_graph():
    np.random.seed(42)
    graph = _make_graph(True)

    runner1 = Evaluator(graph.model, backend="OnnxRuntime")
    pm = PassManager(["remove_zero_channels"])
    graph = pm.optimize(graph, strict=True)

    runner2 = Evaluator(graph.model, backend="OnnxRuntime")

    x = np.random.randn(1, 16, 32, 32).astype(np.float32)
    y0 = runner1(["Z"], {"X": x})[0]
    y1 = runner2(["Z"], {"X": x})[0]

    np.testing.assert_allclose(y0, y1, rtol=1e-2)


def test_remove_zero_channels_on_pruned_graph_with_stopper():
    np.random.seed(42)
    graph = _make_graph(True, True)

    runner1 = Evaluator(graph.model, backend="OnnxRuntime")
    pm = PassManager(["remove_zero_channels"])
    graph = pm.optimize(graph, strict=True)

    runner2 = Evaluator(graph.model, backend="OnnxRuntime")

    x = np.random.randn(1, 16, 32, 32).astype(np.float32)
    y0 = runner1(["Z"], {"X": x})[0]
    y1 = runner2(["Z"], {"X": x})[0]

    np.testing.assert_allclose(y0, y1, rtol=1e-2)
