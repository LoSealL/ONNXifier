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

# pylint: disable=missing-function-docstring

import numpy as np
import onnx
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_type_proto,
    make_value_info,
)
from onnx.numpy_helper import from_array, to_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager


def _build_test_graph1():
    conv0 = make_node("Conv", inputs=["a", "w0"], outputs=["c"], group=4, name="conv0")
    conv1 = make_node("Conv", inputs=["c", "w1"], outputs=["d"], group=1, name="conv1")
    graph = make_graph(
        [conv0, conv1],
        name="graph",
        inputs=[],
        outputs=[make_value_info("d", make_tensor_type_proto(1, [1, 8, 124, 123]))],
        initializer=[
            from_array(np.random.normal(size=[1, 4, 128, 127]).astype("float32"), "a"),
            from_array(np.random.normal(size=[4, 1, 3, 3]).astype("float32"), "w0"),
            from_array(np.random.normal(size=[8, 4, 3, 3]).astype("float32"), "w1"),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    return model


def _build_test_graph2():
    shape = make_node("Shape", inputs=["a"], outputs=["b"], name="shape")
    add = make_node("Add", inputs=["b", "x"], outputs=["c"], name="add")
    graph = make_graph(
        [shape, add],
        name="graph",
        inputs=[make_value_info("a", make_tensor_type_proto(1, [1, 32]))],
        outputs=[
            make_value_info("c", make_tensor_type_proto(onnx.TensorProto.INT64, [2]))
        ],
        initializer=[
            from_array(np.array([32, 1]).astype("int64"), "x"),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    return model


def _build_test_graph3():
    shape = make_node("Shape", inputs=["a"], outputs=["b"], name="shape")
    add = make_node("Add", inputs=["b", "x"], outputs=["c"], name="add")
    graph = make_graph(
        [shape, add],
        name="graph",
        inputs=[make_value_info("a", make_tensor_type_proto(1, ["N", 32]))],
        outputs=[
            make_value_info("c", make_tensor_type_proto(onnx.TensorProto.INT64, [2]))
        ],
        initializer=[
            from_array(np.array([32, 1]).astype("int64"), "x"),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    return model


def _build_test_graph4():
    conv = make_node("Conv", inputs=["x", "w", "b"], outputs=["y"], name="QConv")
    dq_w = make_node(
        "DequantizeLinear",
        inputs=["weight", "w_scale", "w_zp"],
        outputs=["w"],
        name="QWeight",
        axis=0,
    )
    reshape = make_node(
        "Reshape", inputs=["y", "shape"], outputs=["out"], name="Reshape"
    )
    add = make_node("Add", inputs=["p", "q"], outputs=["shape"], name="Add")
    graph = make_graph(
        [conv, dq_w, add, reshape],
        name="graph",
        inputs=[make_value_info("x", make_tensor_type_proto(1, [1, 3, 64, 64]))],
        outputs=[make_value_info("out", make_tensor_type_proto(1, [2, 4 * 64 * 64]))],
        initializer=[
            from_array(np.ones([8, 3, 1, 1], "int8"), "weight"),
            from_array(np.ones([8], "float32"), "w_scale"),
            from_array(np.zeros([8], "int8"), "w_zp"),
            from_array(np.random.normal(0, 0.1, size=[8]).astype("float32"), "b"),
            from_array(np.array([1, 2 * 64 * 64]).astype("int64"), "p"),
            from_array(np.array([1, 2 * 64 * 64]).astype("int64"), "q"),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    return model


def test_fold_constant():
    graph = OnnxGraph(_build_test_graph1())
    passes = PassManager(["fold_constant"])
    folded_graph = passes.optimize(graph, True)
    assert len(folded_graph) == 1


def test_fold_shape():
    graph = OnnxGraph(_build_test_graph2())
    passes = PassManager(["fold_constant"])
    folded_graph = passes.optimize(graph, True)
    assert len(folded_graph) == 1
    for node in folded_graph.nodes.values():
        assert np.all(to_array(node["pb"].attribute[0].t) == [33, 33])


def test_fold_dynamic_shape():
    graph = OnnxGraph(_build_test_graph3())
    passes = PassManager(["fold_constant"])
    folded_graph = passes.optimize(graph, True)
    assert len(folded_graph) == 3


def test_fold_should_bypass_qdq():
    graph = OnnxGraph(_build_test_graph4())
    passes = PassManager(["fold_constant"])
    folded_graph = passes.optimize(graph, True)
    assert any(
        folded_graph.nodes[node]["pb"].op_type == "DequantizeLinear"
        for node in folded_graph
    )
