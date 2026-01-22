"""
Copyright (C) 2024-2025 The ONNXIFIER Authors.

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
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager


def _make_graph():
    gn = make_node(
        "GroupNormalization", ["X", "scale", "bias"], ["Y"], name="GN", num_groups=2
    )
    reshape = make_node("Reshape", ["Y", "shape"], ["Z"], name="reshape")
    add = make_node("Add", ["Z", "add_value"], ["W"], name="add")

    graph = make_graph(
        [gn, reshape, add],
        "test_graph",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 32, 16, 16])],
        [make_tensor_value_info("W", onnx.TensorProto.FLOAT, [1, 32, 256])],
        [
            from_array(np.array([1, 1], np.float32), "scale"),
            from_array(np.array([0, 0], np.float32), "bias"),
            from_array(np.array([1, 32, 256], np.int64), "shape"),
            from_array(np.zeros([256], np.float32), "add_value"),
        ],
    )
    # GroupNormalization is deprecated after opset 18
    model = make_model(
        graph,
        ir_version=9,
        opset_imports=[make_operatorsetid("", 21)],
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_infer_shape_after_gn():
    graph = _make_graph()
    pm = PassManager(["infer_shape"])
    graph = pm.optimize(graph, strict=True)
    assert graph.tensor_shape("Y") == [1, 32, 16, 16]
    assert graph.tensor_shape("Z") == [1, 32, 256]


def _make_graph_sequence():
    split_to = make_node(
        "SplitToSequence",
        ["a", "split"],
        ["seq"],
        name="split",
        keepdims=0,
        axis=1,
    )
    seq_at = make_node(
        "SequenceAt",
        ["seq", "pos"],
        ["seq_out"],
        name="at",
    )
    trans = make_node(
        "Transpose", ["seq_out"], ["seq_out_t"], name="trans", perm=[0, 1]
    )
    reshape = make_node("Reshape", ["seq_out_t", "shape"], ["result"], name="reshape")
    graph = make_graph(
        [split_to, seq_at, trans, reshape],
        "seq_graph",
        [make_tensor_value_info("a", onnx.TensorProto.FLOAT, [2, 3, 4])],
        [make_tensor_value_info("result", onnx.TensorProto.FLOAT, [2, 4])],
        [
            from_array(np.array(1, np.int64), "split"),
            from_array(np.array(0, np.int64), "pos"),
            from_array(np.array([2, 4], np.int64), "shape"),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return model


def test_infer_shape_with_sequence_at():
    model = _make_graph_sequence()
    pm = PassManager(["infer_shape"])
    graph = pm.optimize(OnnxGraph(model), strict=True)

    assert graph.tensor_shape("seq") == [2, 4]
    assert graph.tensor_shape("result") == [2, 4]
    graph._keep_value_info = True
    with pytest.raises(onnx.shape_inference.InferenceError):
        # onnx bugs
        onnx.checker.check_model(graph.model, True)
    # clear the value info to pass the checker
    graph._keep_value_info = False
    graph._value_info_update.clear()
    onnx.checker.check_model(graph.model, True)
