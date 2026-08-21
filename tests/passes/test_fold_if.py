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
    make_tensor,
    make_tensor_type_proto,
    make_value_info,
)
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator

BOOL = onnx.TensorProto.BOOL


def _run(model, feeds=None):
    return Evaluator(model, backend="onnx")(["y"], feeds or {})[0]


def _const_true(name):
    return make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        name=name,
        value=make_tensor(f"{name}_v", onnx.TensorProto.BOOL, [], [True]),
    )


def _branch(op, left, right, output_name):
    g = make_graph(
        [make_node(op, inputs=["t0", "t1"], outputs=[output_name], name="branch_op")],
        name=f"branch_{op}",
        inputs=[make_value_info("cond", make_tensor_type_proto(BOOL, []))],
        outputs=[make_value_info(output_name, make_tensor_type_proto(1, [3]))],
        initializer=[
            from_array(np.array(left, dtype="float32"), "t0"),
            from_array(np.array(right, dtype="float32"), "t1"),
        ],
    )
    return g


def _build_if_graph(cond_value):
    cond = make_node(
        "Constant",
        inputs=[],
        outputs=["cond"],
        name="cond",
        value=make_tensor("cond_v", onnx.TensorProto.BOOL, [], [cond_value]),
    )
    if_node = make_node(
        "If",
        inputs=["cond"],
        outputs=["y"],
        name="if",
        then_branch=_branch("Add", [1, 2, 3], [10, 20, 30], "o"),
        else_branch=_branch("Mul", [1, 2, 3], [10, 20, 30], "o"),
    )
    graph = make_graph(
        [cond, if_node],
        name="graph",
        inputs=[],
        outputs=[make_value_info("y", make_tensor_type_proto(1, [3]))],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    return model


def test_fold_if_then():
    graph = OnnxGraph(_build_if_graph(True))
    folded = PassManager(["fold_if"]).optimize(graph, True)
    assert all(folded.nodes[n]["pb"].op_type != "If" for n in folded)
    assert any(folded.nodes[n]["pb"].op_type == "Add" for n in folded)
    assert "y" in folded.outputs
    assert np.allclose(_run(folded.model), [11, 22, 33])


def test_fold_if_else():
    graph = OnnxGraph(_build_if_graph(False))
    folded = PassManager(["fold_if"]).optimize(graph, True)
    assert all(folded.nodes[n]["pb"].op_type != "If" for n in folded)
    assert any(folded.nodes[n]["pb"].op_type == "Mul" for n in folded)
    assert np.allclose(_run(folded.model), [10, 40, 90])


def test_fold_if_dynamic_cond_untouched():
    # cond comes from a graph input: not statically known, must not fold
    def _plain_branch(op, left, right):
        return make_graph(
            [make_node(op, inputs=["t0", "t1"], outputs=["y"], name=f"branch_{op}")],
            name=f"branch_{op}",
            inputs=[],
            outputs=[make_value_info("y", make_tensor_type_proto(1, [3]))],
            initializer=[
                from_array(np.array(left, dtype="float32"), "t0"),
                from_array(np.array(right, dtype="float32"), "t1"),
            ],
        )

    if_node = make_node(
        "If",
        inputs=["cond"],
        outputs=["y"],
        name="if",
        then_branch=_plain_branch("Add", [1, 2, 3], [10, 20, 30]),
        else_branch=_plain_branch("Mul", [1, 2, 3], [10, 20, 30]),
    )
    graph = make_graph(
        [if_node],
        name="graph",
        inputs=[make_value_info("cond", make_tensor_type_proto(BOOL, []))],
        outputs=[make_value_info("y", make_tensor_type_proto(1, [3]))],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    folded = PassManager(["fold_if"]).optimize(OnnxGraph(model), True)
    assert any(folded.nodes[n]["pb"].op_type == "If" for n in folded)
    assert np.allclose(_run(folded.model, {"cond": np.array(True)}), [11, 22, 33])
    assert np.allclose(_run(folded.model, {"cond": np.array(False)}), [10, 40, 90])


def test_fold_if_outer_scope_capture():
    # branch nodes reference an outer initializer implicitly (no declared input)
    if_node = make_node(
        "If",
        inputs=["cond"],
        outputs=["y"],
        name="if",
        then_branch=make_graph(
            [make_node("Mul", inputs=["w", "w"], outputs=["y"], name="m")],
            name="then",
            inputs=[],
            outputs=[make_value_info("y", make_tensor_type_proto(1, [2]))],
        ),
        else_branch=make_graph(
            [make_node("Add", inputs=["w", "w"], outputs=["y"], name="p")],
            name="else",
            inputs=[],
            outputs=[make_value_info("y", make_tensor_type_proto(1, [2]))],
        ),
    )
    graph = make_graph(
        [_const_true("cond"), if_node],
        name="graph",
        inputs=[],
        outputs=[make_value_info("y", make_tensor_type_proto(1, [2]))],
        initializer=[from_array(np.array([7.0, 8.0], "float32"), "w")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    folded = PassManager(["fold_if"]).optimize(OnnxGraph(model), True)
    assert all(folded.nodes[n]["pb"].op_type != "If" for n in folded)
    assert np.allclose(_run(folded.model), [49.0, 64.0])


def _nested_if_graph(inner_cond_source):
    """Outer If with constant cond; then_branch holds an inner If.

    ``inner_cond_source="input"`` gives the inner If a dynamic condition (model
    input ``x``) so it must survive the fold; ``"outer"`` gives it the outer
    constant condition so it must be folded recursively.
    """
    inner_cond = "x" if inner_cond_source == "input" else "cond"
    inner_if = make_node(
        "If",
        inputs=[inner_cond],
        outputs=["y"],
        name="inner_if",
        then_branch=make_graph(
            [make_node("Add", inputs=["a", "b"], outputs=["inner_y"], name="i_add")],
            name="inner_then",
            inputs=[],
            outputs=[make_value_info("inner_y", make_tensor_type_proto(1, [2]))],
            initializer=[
                from_array(np.array([1.0, 2.0], "float32"), "a"),
                from_array(np.array([3.0, 4.0], "float32"), "b"),
            ],
        ),
        else_branch=make_graph(
            [make_node("Sub", inputs=["a", "b"], outputs=["inner_y"], name="i_sub")],
            name="inner_else",
            inputs=[],
            outputs=[make_value_info("inner_y", make_tensor_type_proto(1, [2]))],
            initializer=[
                from_array(np.array([1.0, 2.0], "float32"), "a"),
                from_array(np.array([3.0, 4.0], "float32"), "b"),
            ],
        ),
    )
    outer_if = make_node(
        "If",
        inputs=["cond"],
        outputs=["y"],
        name="outer_if",
        then_branch=make_graph(
            [inner_if],
            name="outer_then",
            inputs=[],
            outputs=[make_value_info("y", make_tensor_type_proto(1, [2]))],
        ),
        else_branch=make_graph(
            [make_node("Mul", inputs=["a", "b"], outputs=["y"], name="o_mul")],
            name="outer_else",
            inputs=[],
            outputs=[make_value_info("y", make_tensor_type_proto(1, [2]))],
            initializer=[
                from_array(np.array([5.0, 5.0], "float32"), "a"),
                from_array(np.array([6.0, 6.0], "float32"), "b"),
            ],
        ),
    )
    graph = make_graph(
        [_const_true("cond"), outer_if],
        name="graph",
        inputs=(
            [make_value_info("x", make_tensor_type_proto(BOOL, []))]
            if inner_cond_source == "input"
            else []
        ),
        outputs=[make_value_info("y", make_tensor_type_proto(1, [2]))],
    )
    return make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )


def test_fold_if_nested_dynamic_survives():
    # inner If has a dynamic cond: outer folds, inner must stay valid and correct
    model = _nested_if_graph("input")
    folded = PassManager(["fold_if"]).optimize(OnnxGraph(model), True)
    assert not any(folded.nodes[n]["pb"].name == "outer_if" for n in folded), (
        "outer If should be folded"
    )
    for xv in (True, False):
        feeds = {"x": np.array(xv)}
        expected = _run(model, feeds)
        assert np.allclose(_run(folded.model, feeds), expected)


def test_fold_if_nested_constant():
    # inner If cond is the outer constant cond: both must be folded recursively
    model = _nested_if_graph("outer")
    folded = PassManager(["fold_if"]).optimize(OnnxGraph(model), True)
    assert all(folded.nodes[n]["pb"].op_type != "If" for n in folded)
    assert np.allclose(_run(folded.model), [4, 6])  # 1+3, 2+4


def test_fold_if_deterministic_names_and_collision():
    # parent graph already owns a tensor named "dup/mid": the inlined branch
    # tensor "mid" must yield to "dup/mid_1" instead of shadowing it
    then_branch = make_graph(
        [
            make_node("Add", inputs=["t0", "t1"], outputs=["mid"], name="add"),
            make_node("Mul", inputs=["mid", "t0"], outputs=["o"], name="mul"),
        ],
        name="then",
        inputs=[],
        outputs=[make_value_info("o", make_tensor_type_proto(1, [2]))],
        initializer=[
            from_array(np.array([1.0, 2.0], "float32"), "t0"),
            from_array(np.array([10.0, 20.0], "float32"), "t1"),
        ],
    )
    else_branch = make_graph(
        [make_node("Sub", inputs=["t0", "t1"], outputs=["o"], name="sub")],
        name="else",
        inputs=[],
        outputs=[make_value_info("o", make_tensor_type_proto(1, [2]))],
        initializer=[
            from_array(np.array([1.0, 2.0], "float32"), "t0"),
            from_array(np.array([10.0, 20.0], "float32"), "t1"),
        ],
    )
    if_node = make_node(
        "If",
        inputs=["cond"],
        outputs=["y"],
        name="dup",
        then_branch=then_branch,
        else_branch=else_branch,
    )
    graph = make_graph(
        [_const_true("cond"), if_node],
        name="graph",
        inputs=[],
        outputs=[make_value_info("y", make_tensor_type_proto(1, [2]))],
        initializer=[from_array(np.array([0.0, 0.0], "float32"), "dup/mid")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    folded = PassManager(["fold_if"]).optimize(OnnxGraph(model), True)
    onnx.checker.check_model(folded.model)
    assert all(folded.nodes[n]["pb"].op_type != "If" for n in folded)
    outs = [o for n in folded for o in folded.nodes[n]["pb"].output]
    assert "dup/mid_1" in outs
    assert "dup/mid" not in outs  # must not shadow the existing initializer
    assert np.allclose(_run(folded.model), [11.0, 44.0])
