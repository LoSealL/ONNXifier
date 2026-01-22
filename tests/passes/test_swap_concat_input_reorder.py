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
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, PassManager
from onnxifier.evaluator import Evaluator
from onnxifier.graph import OnnxGraph
from onnxifier.passes.swap.swap_concat_input_order import (
    SwapConcatInputOrderRewriter,
)


def _build_graph(group: None | int = None):
    n1 = make_node("Concat", ["a", "b", "c"], ["x"], "cat", axis=1)
    n2 = make_node(
        "Conv",
        ["x", "w", "bias"],
        ["y"],
        "conv",
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        kernel_shape=[3, 3],
        dilations=[1, 1],
        group=group,
    )
    g = group or 1
    graph = make_graph(
        [n1, n2],
        "graph",
        [
            make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 4, 4]),
            make_tensor_value_info("b", TensorProto.FLOAT, [1, 5, 4, 4]),
            make_tensor_value_info("c", TensorProto.FLOAT, [1, 8, 4, 4]),
        ],
        [make_tensor_value_info("y", TensorProto.FLOAT, [1, 16, 4, 4])],
        [
            from_array(np.random.randn(16, 16 // g, 3, 3).astype("float32"), name="w"),
            from_array(np.random.randn(16).astype("float32"), name="bias"),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_swap_concat_input_order():
    graph = _build_graph()
    runner1 = Evaluator(graph.model)
    pm = PassManager(
        ["swap_concat_input_order"],
        configs={"swap_concat_input_order": {"order": [2, 1, 0]}},
    )

    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model)

    a = np.random.randn(1, 3, 4, 4).astype("float32")
    b = np.random.randn(1, 5, 4, 4).astype("float32")
    c = np.random.randn(1, 8, 4, 4).astype("float32")

    try:
        y1 = runner1(["y"], {"a": a, "b": b, "c": c})[0]
        y2 = runner2(["y"], {"a": a, "b": b, "c": c})[0]
        np.testing.assert_allclose(y1, y2, rtol=1e-4)
    except Exception:
        graph.save("test_swap_concat_input_order_failed.onnx", check=False)
        raise


def test_with_group_conv():
    graph = _build_graph(group=2)
    rewriter = SwapConcatInputOrderRewriter()
    with pytest.raises(ValueError):
        rewriter(graph, order=[2, 1, 0])


def test_with_error_order():
    graph = _build_graph()
    rewriter = SwapConcatInputOrderRewriter()
    with pytest.raises(ValueError):
        rewriter(graph, order=[1, 0])

    with pytest.raises(ValueError):
        rewriter(graph)
