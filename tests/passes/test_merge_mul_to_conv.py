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
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, PassManager
from onnxifier.evaluator import Evaluator
from onnxifier.graph import OnnxGraph


def _build_graph(elt_type: str = "Add"):
    mul = make_node("Mul", ["x", "m1"], ["o1"], "mul1")
    elt = make_node(elt_type, ["o1", "a2"], ["o2"], "elt2")
    conv = make_node(
        "Conv", ["o2", "w3", "b3"], ["y"], "conv3", kernel_shape=[2, 2], strides=[2, 2]
    )
    graph = make_graph(
        [mul, elt, conv],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [make_value_info("y", make_tensor_type_proto(1, [1, 6, 128, 128]))],
        [
            make_tensor(
                "m1", 1, [1, 3, 1, 1], np.random.rand(1, 3, 1, 1).astype("float32")
            ),
            make_tensor(
                "a2", 1, [1, 3, 1, 1], np.random.rand(1, 3, 1, 1).astype("float32")
            ),
            make_tensor(
                "w3", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
            make_tensor(
                "b3", 1, [6, 1, 1, 1], np.random.rand(6, 1, 1, 1).astype("float32")
            ),
        ],
    )
    return make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )


def _build_graph2():
    mul = make_node("Mul", ["y", "m1"], ["z"], "mul1")
    conv = make_node(
        "Conv", ["x", "w3", "b3"], ["y"], "conv3", kernel_shape=[2, 2], strides=[2, 2]
    )
    graph = make_graph(
        [conv, mul],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [make_value_info("z", make_tensor_type_proto(1, [1, 6, 128, 128]))],
        [
            make_tensor(
                "m1", 1, [1, 6, 1, 1], np.random.rand(1, 6, 1, 1).astype("float32")
            ),
            make_tensor(
                "w3", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
            make_tensor(
                "b3", 1, [6, 1, 1, 1], np.random.rand(6, 1, 1, 1).astype("float32")
            ),
        ],
    )
    return make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )


def test_merge_mul_to_conv():
    model = _build_graph()
    runner1 = Evaluator(model)
    x = np.random.uniform(0, 1, size=[1, 3, 256, 256]).astype(np.float32)
    y1 = runner1(None, {"x": x})[0]

    graph = OnnxGraph(model)
    pm = PassManager(["merge_mul_to_conv", "onnxsim"])
    graph = pm.optimize(graph, strict=True)

    # check output
    runner2 = Evaluator(graph.model)
    y2 = runner2(None, {"x": x})[0]
    assert np.allclose(y1, y2)

    # check conv number
    mul_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "Mul":
            mul_number += 1
    assert mul_number == 0


def test_merge_mul_to_conv2():
    model = _build_graph("Sub")
    runner1 = Evaluator(model)
    x = np.random.uniform(0, 1, size=[1, 3, 256, 256]).astype(np.float32)
    y1 = runner1(None, {"x": x})[0]

    graph = OnnxGraph(model)
    pm = PassManager(["merge_mul_to_conv", "onnxsim"])
    graph = pm.optimize(graph, strict=True)

    # check output
    runner2 = Evaluator(graph.model)
    y2 = runner2(None, {"x": x})[0]
    assert np.allclose(y1, y2, atol=1e-6), f"{np.abs(y1 - y2).max()}"

    # check conv number
    mul_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "Mul":
            mul_number += 1
    assert mul_number == 0


def test_merge_mul_to_conv_pattern2():
    model = _build_graph2()
    runner1 = Evaluator(model)
    x = np.random.uniform(0, 1, size=[1, 3, 256, 256]).astype(np.float32)
    y1 = runner1(None, {"x": x})[0]

    graph = OnnxGraph(model)
    pm = PassManager(["merge_mul_to_conv", "onnxsim"])
    graph = pm.optimize(graph, strict=True)

    runner2 = Evaluator(graph.model)
    y2 = runner2(None, {"x": x})[0]
    assert np.allclose(y1, y2, atol=1e-6), f"{np.abs(y1 - y2).max()}"

    # Check if Mul is merged
    mul_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "Mul":
            mul_number += 1
    assert mul_number == 0


def test_merge_mul_to_conv_unsupported_elt_type():
    """Test Pattern1 with unsupported element type (should not optimize)"""
    mul = make_node("Mul", ["x", "m1"], ["o1"], "mul1")
    relu = make_node("Relu", ["o1"], ["o2"], "relu2")  # Unsupported type
    conv = make_node(
        "Conv", ["o2", "w3", "b3"], ["y"], "conv3", kernel_shape=[2, 2], strides=[2, 2]
    )
    graph = make_graph(
        [mul, relu, conv],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [make_value_info("y", make_tensor_type_proto(1, [1, 6, 128, 128]))],
        [
            make_tensor(
                "m1", 1, [1, 3, 1, 1], np.random.rand(1, 3, 1, 1).astype("float32")
            ),
            make_tensor(
                "w3", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
            make_tensor(
                "b3", 1, [6, 1, 1, 1], np.random.rand(6, 1, 1, 1).astype("float32")
            ),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )

    graph = OnnxGraph(model)
    pm = PassManager(["merge_mul_to_conv", "onnxsim"])
    graph = pm.optimize(graph, strict=True)

    # Check that Mul should still exist (not optimized)
    mul_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "Mul":
            mul_number += 1
    assert mul_number == 1


def test_merge_mul_to_conv_no_constant():
    """Test Pattern1 with no constant in mul (should not optimize)"""
    mul = make_node("Mul", ["x", "y"], ["o1"], "mul1")  # No constant
    add = make_node("Add", ["o1", "a2"], ["o2"], "add2")
    conv = make_node(
        "Conv", ["o2", "w3", "b3"], ["z"], "conv3", kernel_shape=[2, 2], strides=[2, 2]
    )
    graph = make_graph(
        [mul, add, conv],
        "graph",
        [
            make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256])),
            make_value_info("y", make_tensor_type_proto(1, [1, 3, 256, 256])),
        ],
        [make_value_info("z", make_tensor_type_proto(1, [1, 6, 128, 128]))],
        [
            make_tensor(
                "a2", 1, [1, 3, 1, 1], np.random.rand(1, 3, 1, 1).astype("float32")
            ),
            make_tensor(
                "w3", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
            make_tensor(
                "b3", 1, [6, 1, 1, 1], np.random.rand(6, 1, 1, 1).astype("float32")
            ),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )

    graph = OnnxGraph(model)
    pm = PassManager(["merge_mul_to_conv", "onnxsim"])
    graph = pm.optimize(graph, strict=True)

    # Check that Mul should still exist (not optimized)
    mul_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "Mul":
            mul_number += 1
    assert mul_number == 1


def test_merge_mul_to_conv_multi_fanout():
    """Test Pattern2 with multi fan-out (should not optimize)"""
    conv = make_node(
        "Conv", ["x", "w"], ["y"], "conv1", kernel_shape=[2, 2], strides=[2, 2]
    )
    mul1 = make_node("Mul", ["y", "m1"], ["z1"], "mul1")
    mul2 = make_node("Mul", ["y", "m2"], ["z2"], "mul2")  # Multi fan-out
    graph = make_graph(
        [conv, mul1, mul2],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [
            make_value_info("z1", make_tensor_type_proto(1, [1, 6, 128, 128])),
            make_value_info("z2", make_tensor_type_proto(1, [1, 6, 128, 128])),
        ],
        [
            make_tensor(
                "w", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
            make_tensor(
                "m1", 1, [1, 6, 1, 1], np.random.rand(1, 6, 1, 1).astype("float32")
            ),
            make_tensor(
                "m2", 1, [1, 6, 1, 1], np.random.rand(1, 6, 1, 1).astype("float32")
            ),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )

    graph = OnnxGraph(model)
    pm = PassManager(["merge_mul_to_conv", "onnxsim"])
    graph = pm.optimize(graph, strict=True)

    # Check that Mul should still exist (not optimized)
    mul_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "Mul":
            mul_number += 1
    assert mul_number == 2


def test_merge_mul_to_conv_cannot_broadcast():
    """Test Pattern2 with factor that cannot broadcast to weight"""
    conv = make_node(
        "Conv", ["x", "w"], ["y"], "conv1", kernel_shape=[2, 2], strides=[2, 2]
    )
    mul = make_node("Mul", ["y", "m"], ["z"], "mul1")
    graph = make_graph(
        [conv, mul],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [make_value_info("z", make_tensor_type_proto(1, [1, 6, 128, 128]))],
        [
            make_tensor(
                "w", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
            make_tensor(
                "m",
                1,
                [1, 6, 128, 128],  # Cannot broadcast (3 > 1 in first dims)
                np.random.rand(1, 6, 128, 128).astype("float32"),
            ),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )

    graph = OnnxGraph(model)
    pm = PassManager(["merge_mul_to_conv", "onnxsim"])
    graph = pm.optimize(graph, strict=True)

    # Check that Mul should still exist (not optimized)
    mul_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "Mul":
            mul_number += 1
    assert mul_number == 1


def test_merge_mul_to_conv_no_bias():
    """Test Pattern2 with Conv without bias"""
    conv = make_node(
        "Conv", ["x", "w"], ["y"], "conv1", kernel_shape=[2, 2], strides=[2, 2]
    )
    mul = make_node("Mul", ["m", "y"], ["z"], "mul1")
    graph = make_graph(
        [conv, mul],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [make_value_info("z", make_tensor_type_proto(1, [1, 6, 128, 128]))],
        [
            make_tensor(
                "w", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
            make_tensor(
                "m", 1, [1, 6, 1, 1], np.random.rand(1, 6, 1, 1).astype("float32")
            ),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )

    runner1 = Evaluator(model)
    x = np.random.uniform(0, 1, size=[1, 3, 256, 256]).astype(np.float32)
    y1 = runner1(None, {"x": x})[0]

    graph = OnnxGraph(model)
    pm = PassManager(["merge_mul_to_conv", "onnxsim"])
    graph = pm.optimize(graph, strict=True)

    runner2 = Evaluator(graph.model)
    y2 = runner2(None, {"x": x})[0]
    assert np.allclose(y1, y2, atol=1e-6), f"{np.abs(y1 - y2).max()}"

    # Check that Mul should be merged
    mul_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "Mul":
            mul_number += 1
    assert mul_number == 0
