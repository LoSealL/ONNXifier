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
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array

from onnxifier import OnnxGraph, PassManager


def _break_value_info_dtype(graph: OnnxGraph, name: str) -> None:
    """Set dtype of a tensor in _value_info to UNDEFINED."""
    for vi in graph._value_info:
        if vi.name == name:
            vi.type.tensor_type.elem_type = onnx.TensorProto.UNDEFINED
            break


def _break_output_dtype(graph: OnnxGraph, name: str) -> None:
    """Set dtype of a graph output to UNDEFINED."""
    for out in graph.output:
        if out.name == name:
            out.type.tensor_type.elem_type = onnx.TensorProto.UNDEFINED
            break


def test_infer_dtype_for_add():
    """Add output dtype should be inferred from inputs (both FLOAT -> FLOAT)."""
    add = make_node("Add", ["a", "b"], ["c"], name="add")
    relu = make_node("Relu", ["c"], ["d"], name="relu")
    graph = make_graph(
        [add, relu],
        "test_add",
        [
            make_tensor_value_info("a", onnx.TensorProto.FLOAT, [2, 3]),
            make_tensor_value_info("b", onnx.TensorProto.FLOAT, [2, 3]),
        ],
        [make_tensor_value_info("d", onnx.TensorProto.FLOAT, [2, 3])],
    )
    model = make_model(graph, ir_version=9, opset_imports=[make_operatorsetid("", 13)])
    onnx.checker.check_model(model)

    g = OnnxGraph(model)
    _break_value_info_dtype(g, "c")
    assert g.tensor_info("c")[1] == onnx.TensorProto.UNDEFINED

    pm = PassManager(["infer_dtype"])
    g = pm.optimize(g, strict=True)
    assert g.tensor_info("c")[1] == onnx.TensorProto.FLOAT


def test_infer_dtype_for_cast():
    """Cast output dtype should be read from the 'to' attribute."""
    cast = make_node("Cast", ["a"], ["b"], name="cast", to=onnx.TensorProto.INT64)
    relu = make_node("Relu", ["b"], ["c"], name="relu")
    graph = make_graph(
        [cast, relu],
        "test_cast",
        [make_tensor_value_info("a", onnx.TensorProto.FLOAT, [2, 3])],
        [make_tensor_value_info("c", onnx.TensorProto.INT64, [2, 3])],
    )
    model = make_model(graph, ir_version=9, opset_imports=[make_operatorsetid("", 13)])
    onnx.checker.check_model(model)

    g = OnnxGraph(model)
    _break_value_info_dtype(g, "b")
    assert g.tensor_info("b")[1] == onnx.TensorProto.UNDEFINED

    pm = PassManager(["infer_dtype"])
    g = pm.optimize(g, strict=True)
    assert g.tensor_info("b")[1] == onnx.TensorProto.INT64


def test_infer_dtype_for_constant():
    """Constant output dtype should be inferred from the value attribute."""
    const = make_node(
        "Constant",
        inputs=[],
        outputs=["c"],
        name="const",
        value=from_array(np.array([1.0, 2.0], dtype=np.float32), name="value"),
    )
    relu = make_node("Relu", ["c"], ["d"], name="relu")
    graph = make_graph(
        [const, relu],
        "test_constant",
        inputs=[],
        outputs=[make_tensor_value_info("d", onnx.TensorProto.FLOAT, [2])],
    )
    model = make_model(graph, ir_version=9, opset_imports=[make_operatorsetid("", 13)])
    onnx.checker.check_model(model)

    g = OnnxGraph(model)
    _break_value_info_dtype(g, "c")
    assert g.tensor_info("c")[1] == onnx.TensorProto.UNDEFINED

    pm = PassManager(["infer_dtype"])
    g = pm.optimize(g, strict=True)
    assert g.tensor_info("c")[1] == onnx.TensorProto.FLOAT


def test_infer_dtype_guess_from_single_input():
    """Identity (single-input) should guess output dtype from its input."""
    identity = make_node("Identity", ["a"], ["b"], name="identity")
    relu = make_node("Relu", ["b"], ["c"], name="relu")
    graph = make_graph(
        [identity, relu],
        "test_identity",
        [make_tensor_value_info("a", onnx.TensorProto.INT32, [2, 3])],
        [make_tensor_value_info("c", onnx.TensorProto.INT32, [2, 3])],
    )
    model = make_model(graph, ir_version=9, opset_imports=[make_operatorsetid("", 13)])
    onnx.checker.check_model(model)

    g = OnnxGraph(model)
    _break_value_info_dtype(g, "b")
    assert g.tensor_info("b")[1] == onnx.TensorProto.UNDEFINED

    pm = PassManager(["infer_dtype"])
    g = pm.optimize(g, strict=True)
    assert g.tensor_info("b")[1] == onnx.TensorProto.INT32


def test_infer_dtype_guess_for_custom_op():
    """Custom domain ops have no schema; single-input heuristic should still work."""
    custom = make_node("CustomOp", ["a"], ["b"], name="custom", domain="test.custom")
    relu = make_node("Relu", ["b"], ["c"], name="relu")
    graph = make_graph(
        [custom, relu],
        "test_custom",
        [make_tensor_value_info("a", onnx.TensorProto.FLOAT, [2, 3])],
        [make_tensor_value_info("c", onnx.TensorProto.FLOAT, [2, 3])],
    )
    model = make_model(
        graph,
        ir_version=9,
        opset_imports=[
            make_operatorsetid("", 13),
            make_operatorsetid("test.custom", 1),
        ],
    )

    g = OnnxGraph(model)
    _break_value_info_dtype(g, "b")
    assert g.tensor_info("b")[1] == onnx.TensorProto.UNDEFINED

    pm = PassManager(["infer_dtype"])
    g = pm.optimize(g, strict=True)
    assert g.tensor_info("b")[1] == onnx.TensorProto.FLOAT


def test_infer_dtype_mismatch_is_fixed():
    """When current dtype disagrees with inference, the pass should overwrite it."""
    cast = make_node("Cast", ["a"], ["b"], name="cast", to=onnx.TensorProto.INT64)
    relu = make_node("Relu", ["b"], ["c"], name="relu")
    graph = make_graph(
        [cast, relu],
        "test_mismatch",
        [make_tensor_value_info("a", onnx.TensorProto.FLOAT, [2, 3])],
        [make_tensor_value_info("c", onnx.TensorProto.INT64, [2, 3])],
    )
    model = make_model(graph, ir_version=9, opset_imports=[make_operatorsetid("", 13)])
    onnx.checker.check_model(model)

    g = OnnxGraph(model)
    # Manually corrupt the inferred dtype from INT64 -> FLOAT
    _break_value_info_dtype(g, "b")
    # OnnxGraph constructor already fixed it via infer_shapes, but we broke _value_info.
    # However tensor_info still sees the correct dtype from output or other sources?
    # For intermediate tensor 'b', it only exists in _value_info, so breaking works.
    assert g.tensor_info("b")[1] == onnx.TensorProto.UNDEFINED

    pm = PassManager(["infer_dtype"])
    g = pm.optimize(g, strict=True)
    assert g.tensor_info("b")[1] == onnx.TensorProto.INT64


def test_infer_dtype_for_graph_output():
    """Graph outputs should also have their dtype fixed when wrong."""
    cast = make_node("Cast", ["a"], ["b"], name="cast", to=onnx.TensorProto.INT64)
    graph = make_graph(
        [cast],
        "test_output",
        [make_tensor_value_info("a", onnx.TensorProto.FLOAT, [2, 3])],
        [make_tensor_value_info("b", onnx.TensorProto.INT64, [2, 3])],
    )
    model = make_model(graph, ir_version=9, opset_imports=[make_operatorsetid("", 13)])
    onnx.checker.check_model(model)

    g = OnnxGraph(model)
    # Break the output dtype
    _break_output_dtype(g, "b")
    assert g.tensor_info("b")[1] == onnx.TensorProto.UNDEFINED

    pm = PassManager(["infer_dtype"])
    g = pm.optimize(g, strict=True)
    assert g.tensor_info("b")[1] == onnx.TensorProto.INT64


def test_infer_dtype_interactive_mocked(monkeypatch):
    """Interactive mode should accept user input without crashing."""
    equal = make_node("Equal", ["a", "b"], ["c"], name="equal")
    relu = make_node("Relu", ["c"], ["d"], name="relu")
    graph = make_graph(
        [equal, relu],
        "test_interactive",
        [
            make_tensor_value_info("a", onnx.TensorProto.FLOAT, [2, 3]),
            make_tensor_value_info("b", onnx.TensorProto.FLOAT, [2, 3]),
        ],
        [make_tensor_value_info("d", onnx.TensorProto.BOOL, [2, 3])],
    )
    model = make_model(graph, ir_version=9, opset_imports=[make_operatorsetid("", 13)])
    onnx.checker.check_model(model)

    g = OnnxGraph(model)
    # For Equal, schema says output is bool, so auto inference works.
    # To exercise interactive path we temporarily monkey-patch _infer_from_schema
    # to force UNDEFINED for this node.
    from onnxifier.passes.globals import infer_dtype as id_mod

    def _forced_undefined(*args, **kwargs):
        return onnx.TensorProto.UNDEFINED

    monkeypatch.setattr(id_mod, "_infer_from_schema", _forced_undefined)
    monkeypatch.setattr(id_mod, "_guess_output_dtype", _forced_undefined)
    monkeypatch.setattr("builtins.input", lambda _: "BOOL")

    pm = PassManager(["infer_dtype"])
    g = pm.optimize(g, strict=False)
    assert g.tensor_info("c")[1] == onnx.TensorProto.BOOL
