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
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier import ONNXIFIER_OPSET, PassManager
from onnxifier.graph import OnnxGraph


def _build_graph1():
    """Conv-Relu-Add-Mul-AveragePool-Softmax"""
    conv = make_node("Conv", ["inputs", "w"], ["conv_out"], "conv", kernel_shape=[1, 1])
    relu = make_node("Relu", ["conv_out"], ["relu_out"], "relu")
    add = make_node("Add", ["relu_out", "b"], ["add_out"], "add")
    mul = make_node("Mul", ["add_out", "c"], ["mul_out"], "mul")
    avgpool = make_node(
        "AveragePool",
        ["mul_out"],
        ["avgpool_out"],
        "avgpool",
        kernel_shape=[1, 1, 192, 192],
    )
    softmax = make_node("Softmax", ["avgpool_out"], ["softmax_out"], "softmax", axis=1)
    graph = make_graph(
        [conv, relu, add, mul, avgpool, softmax],
        "graph",
        [make_value_info("inputs", make_tensor_type_proto(1, [1, 3, 192, 192]))],
        [make_value_info("softmax_out", make_tensor_type_proto(1, [1, 32, 1, 1]))],
        [
            numpy_helper.from_array(
                np.random.randn(32, 3, 1, 1).astype(np.float32), "w"
            ),
            numpy_helper.from_array(
                np.random.randn(1, 32, 1, 1).astype(np.float32), "b"
            ),
            numpy_helper.from_array(
                np.random.randn(1, 32, 1, 1).astype(np.float32), "c"
            ),
        ],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])
    onnx.checker.check_model(model)
    return model


def _build_graph2():
    """
    Conv-Sigmoid-
       |________|-Add
    """
    conv = make_node("Conv", ["inputs", "w"], ["conv_out"], "conv", kernel_shape=[1, 1])
    sigmoid = make_node("Sigmoid", ["conv_out"], ["sigmoid_out"], "sigmoid")
    add = make_node("Add", ["sigmoid_out", "conv_out"], ["add_out"], "add")
    graph = make_graph(
        [conv, sigmoid, add],
        "graph",
        [make_value_info("inputs", make_tensor_type_proto(1, [1, 3, 192, 192]))],
        [make_value_info("add_out", make_tensor_type_proto(1, [1, 32, 192, 192]))],
        [numpy_helper.from_array(np.random.randn(32, 3, 1, 1).astype(np.float32), "w")],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])
    onnx.checker.check_model(model)
    return model


def test_split_but_not_activation_functions():
    graph = OnnxGraph(_build_graph1())
    pm = PassManager(["split_non_compute_outputs"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 2  # Conv-Relu
    assert "conv" in graph
    assert "relu" in graph
    assert graph.output[0].name == "relu_out"


def test_split_diamond_pattern():
    graph = OnnxGraph(_build_graph2())
    pm = PassManager(["split_non_compute_outputs"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 1  # Conv
    assert "conv" in graph
    assert len(graph.outputs) == 1
