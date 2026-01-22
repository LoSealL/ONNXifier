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
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_value_info,
)

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.passes.utils import evaluate_on_node


def _build_test_graph():
    # Create a constant tensor as input
    input_tensor = make_tensor(
        "input_value",
        onnx.TensorProto.FLOAT,
        [2, 3, 4],
        np.random.rand(2, 3, 4).astype(np.float32),
    )

    # Create a Shape node that takes the input
    shape_node = make_node("Shape", ["input"], ["shape_output"], name="shape_node")

    # Build the graph
    graph = make_graph(
        [shape_node],
        "shape_test_graph",
        [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3, 4])],
        [make_tensor_value_info("shape_output", onnx.TensorProto.INT64, [3])],
        initializer=[input_tensor],
    )

    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, True)
    return model


def _build_test_graph_with_attrs():
    # Create a constant tensor as input
    input_tensor = make_tensor(
        "input_value",
        onnx.TensorProto.FLOAT,
        [2, 3, 4, 5],
        np.random.rand(2, 3, 4, 5).astype(np.float32),
    )

    # Create a Shape node with start and end attributes
    shape_node = make_node(
        "Shape", ["input"], ["shape_output"], name="shape_node", start=1, end=3
    )

    # Build the graph
    graph = make_graph(
        [shape_node],
        "shape_attr_test_graph",
        [make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3, 4, 5])],
        [make_tensor_value_info("shape_output", onnx.TensorProto.INT64, [2])],
        initializer=[input_tensor],
    )

    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, True)
    return model


def test_shape_to_constant_basic():
    model = _build_test_graph()
    graph = OnnxGraph(model)
    pm = PassManager(["shape_to_constant"])
    graph = pm.optimize(graph, strict=True)

    # After optimization, there should be no Shape node
    assert len(graph.nodes) == 1
    for node_name in graph.nodes:
        node = graph.nodes[node_name]["pb"]
        # The Shape node should be replaced with Constant
        assert node.op_type == "Constant"
        # Check if the constant values are correct ([2, 3, 4])
        const_value = evaluate_on_node(graph, node)
        assert const_value is not None
        np.testing.assert_array_equal(const_value, np.array([2, 3, 4], dtype=np.int64))


def test_shape_to_constant_with_attributes():
    model = _build_test_graph_with_attrs()
    graph = OnnxGraph(model)
    pm = PassManager(["shape_to_constant"])
    graph = pm.optimize(graph, strict=True)

    # After optimization, there should be no Shape node
    assert len(graph.nodes) == 1
    for node_name in graph.nodes:
        node = graph.nodes[node_name]["pb"]
        # The Shape node should be replaced with Constant
        assert node.op_type == "Constant"
        # Check if the constant values are correct ([3, 4]) - sliced with start=1, end=3
        const_value = evaluate_on_node(graph, node)
        assert const_value is not None
        np.testing.assert_array_equal(const_value, np.array([3, 4], dtype=np.int64))
