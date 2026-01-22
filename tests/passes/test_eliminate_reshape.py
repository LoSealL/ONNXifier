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
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager


def _build_graph1():
    relu = make_node("Relu", ["x"], ["relu_out"])
    reshape1 = make_node("Reshape", ["relu_out", "shape1"], ["r1_out"])
    reshape2 = make_node("Reshape", ["r1_out", "shape2"], ["r2_out"])
    graph = make_graph(
        [relu, reshape1, reshape2],
        "test_graph",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2, 3])],
        [make_tensor_value_info("r2_out", onnx.TensorProto.FLOAT, [6])],
        [
            onnx.numpy_helper.from_array(np.array([3, 2], np.int64), "shape1"),
            onnx.numpy_helper.from_array(np.array([6], np.int64), "shape2"),
        ],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    onnx.checker.check_model(model)
    return model


def _build_graph2():
    reshape1 = make_node("Reshape", ["x", "shape1"], ["r1_out"])
    reshape2 = make_node("Reshape", ["r1_out", "shape2"], ["r2_out"])
    relu = make_node("Relu", ["r2_out"], ["relu_out"])
    graph = make_graph(
        [reshape1, reshape2, relu],
        "test_graph",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2, 3])],
        [make_tensor_value_info("relu_out", onnx.TensorProto.FLOAT, [6])],
        [
            onnx.numpy_helper.from_array(np.array([3, 2], np.int64), "shape1"),
            onnx.numpy_helper.from_array(np.array([6], np.int64), "shape2"),
        ],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    onnx.checker.check_model(model)
    return model


def test_eliminate_duplicated_reshape():
    graph = OnnxGraph(_build_graph1())
    pm = PassManager(["onnxsim"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 2
    assert graph.output[0].name == "r2_out"
    assert graph.tensor_shape("r2_out") == [6]


def test_eliminate_tail_reshape():
    graph = OnnxGraph(_build_graph1())
    pm = PassManager(["onnxsim", "eliminate_reshape"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 1
    assert graph.output[0].name == "r2_out"
    assert graph.tensor_shape("r2_out") == [1, 2, 3]


def test_eliminate_head_reshape():
    graph = OnnxGraph(_build_graph2())
    pm = PassManager(["onnxsim", "eliminate_reshape"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 1
    assert graph.input[0].name == "x"
    assert graph.tensor_shape("x") == [6]
