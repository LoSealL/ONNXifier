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
from onnxscript import script
from onnxscript.onnx_opset import opset23 as op

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.domain.shape_inference import register_shape_inference


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


@register_shape_inference("test", "myop1")
@script()
def myop1(a, b, c):
    shape_a = op.Shape(a)
    shape_b = op.Shape(b)
    shape_c = op.Shape(c)
    return op.CastLike(op.ConstantOfShape(shape_a + shape_b + shape_c), a)


@register_shape_inference("test", "myop2")
@script()
def myop2(a, attr1: int = 1):
    return op.Unsqueeze(a, axes=attr1)


def test_infer_shape_with_custom_op1():
    nodes = [
        onnx.helper.make_node(
            "myop1", ["a", "b", "c"], ["d"], name="op1", domain="test"
        ),
        onnx.helper.make_node(
            "myop1", ["d", "b", "c"], ["x"], name="op2", domain="test"
        ),
    ]
    model = onnx.helper.make_model(
        graph=onnx.helper.make_graph(
            nodes,
            "test",
            [
                onnx.helper.make_tensor_value_info("a", 1, [1, 2, 3]),
                onnx.helper.make_tensor_value_info("b", 2, [1, 2, 3]),
                onnx.helper.make_tensor_value_info("c", 3, [1, 2, 3]),
            ],
            [
                onnx.helper.make_tensor_value_info("x", 0, ["a", "b", "c"]),
            ],
        ),
        ir_version=onnx.IR_VERSION,
        opset_imports=[onnx.helper.make_operatorsetid("test", 1)],
    )
    graph = OnnxGraph(model)
    pm = PassManager(["infer_shape"])
    graph = pm.optimize(graph, strict=True)
    assert tuple(graph.tensor_shape("a")) == (1, 2, 3)
    assert tuple(graph.tensor_shape("b")) == (1, 2, 3)
    assert tuple(graph.tensor_shape("c")) == (1, 2, 3)
    assert tuple(graph.tensor_shape("d")) == (3, 6, 9)
    assert tuple(graph.tensor_shape("x")) == (5, 10, 15)
    assert graph.tensor_type("d") == 1
    assert graph.tensor_type("x") == 1


def test_infer_shape_with_custom_attr():
    nodes = [
        onnx.helper.make_node(
            "myop2", ["a"], ["b"], name="op1", domain="test", attr1=1
        ),
        onnx.helper.make_node(
            "myop2", ["b"], ["c"], name="op2", domain="test", attr1=2
        ),
    ]
    model = onnx.helper.make_model(
        graph=onnx.helper.make_graph(
            nodes,
            "test",
            [
                onnx.helper.make_tensor_value_info("a", 1, [1, 2, 3]),
            ],
            [
                onnx.helper.make_tensor_value_info("c", 0, ["a", "b", "c", "d", "e"]),
            ],
        ),
        ir_version=onnx.IR_VERSION,
        opset_imports=[onnx.helper.make_operatorsetid("test", 1)],
    )
    graph = OnnxGraph(model)
    pm = PassManager(["infer_shape"])
    graph = pm.optimize(graph, strict=True)
    assert tuple(graph.tensor_shape("b")) == (1, 1, 2, 3)
    assert tuple(graph.tensor_shape("c")) == (1, 1, 1, 2, 3)
    assert graph.tensor_type("b") == 1
    assert graph.tensor_type("c") == 1


def test_no_temp_functions_leaked():
    """Verify temporary functions are cleaned up after inference."""
    a = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [1, 2, 3])
    b = onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [1, 2, 3])
    c = onnx.helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT, [1, 2, 3])
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "myop1",
        ["a", "b", "c"],
        ["x"],
        domain="test",
    )

    graph = onnx.helper.make_graph([node], "test", [a, b, c], [x])
    model = onnx.helper.make_model(graph)
    model.opset_import.append(onnx.helper.make_opsetid("test", 1))

    pm = PassManager(["infer_shape"])
    graph_with_shapes = pm.optimize(OnnxGraph(model), strict=True)

    # Verify no temporary functions in output model
    assert len(graph_with_shapes.model.functions) == 0


def test_multiple_custom_ops():
    """Test shape inference with multiple custom ops."""
    a = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [1, 2, 3])
    b = onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [1, 2, 3])
    c = onnx.helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT, [1, 2, 3])
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, None)

    node1 = onnx.helper.make_node(
        "myop1",
        ["a", "b", "c"],
        ["d"],
        domain="test",
    )
    node2 = onnx.helper.make_node(
        "myop1",
        ["d", "b", "c"],
        ["x"],
        domain="test",
    )

    graph = onnx.helper.make_graph([node1, node2], "test", [a, b, c], [x])
    model = onnx.helper.make_model(graph)
    model.opset_import.append(onnx.helper.make_opsetid("test", 1))

    pm = PassManager(["infer_shape"])
    graph_with_shapes = pm.optimize(OnnxGraph(model), strict=True)

    # d is intermediate, use tensor_info to check
    d_shape, _ = graph_with_shapes.tensor_info("d")
    assert d_shape == [3, 6, 9]
    assert tuple(graph_with_shapes.tensor_shape("x")) == (5, 10, 15)


def test_mixed_registered_unregistered_domain_ops():
    """Test mixed registered and unregistered domain ops."""
    a = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [1, 2, 3])
    b = onnx.helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [1, 2, 3])
    c = onnx.helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT, [1, 2, 3])
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, None)

    # Registered custom op
    node1 = onnx.helper.make_node(
        "myop1",
        ["a", "b", "c"],
        ["d"],
        domain="test",
    )
    # Unregistered domain op
    node2 = onnx.helper.make_node("UnknownOp", ["d"], ["x"], domain="unregistered")

    graph = onnx.helper.make_graph([node1, node2], "test", [a, b, c], [x])
    model = onnx.helper.make_model(graph)
    model.opset_import.append(onnx.helper.make_opsetid("test", 1))
    model.opset_import.append(onnx.helper.make_opsetid("unregistered", 1))

    pm = PassManager(["infer_shape"])
    graph_with_shapes = pm.optimize(OnnxGraph(model), strict=True)

    # Registered op should have inferred shape
    d_shape, _ = graph_with_shapes.tensor_info("d")
    assert d_shape == [3, 6, 9]
    # Unregistered op should have unknown shape
    x_shape, _ = graph_with_shapes.tensor_info("x")
    assert x_shape is None


def test_cleanup_on_inference_failure():
    """Test temporary functions are cleaned up even on failure."""
    a = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [1, 2, 3])
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "myop1",
        ["a", "", ""],
        ["x"],
        domain="test",
    )

    graph = onnx.helper.make_graph([node], "test", [a], [x])
    model = onnx.helper.make_model(graph)
    model.opset_import.append(onnx.helper.make_opsetid("test", 1))

    # Should not raise even if inference partially fails
    pm = PassManager(["infer_shape"])
    graph_with_shapes = pm.optimize(OnnxGraph(model), strict=True)

    # Verify no temporary functions leaked
    assert len(graph_with_shapes.model.functions) == 0
