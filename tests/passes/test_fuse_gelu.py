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

from onnxifier import ONNXIFIER_IR_VERSION, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _make_graph(
    half_value: float = 0.5,
    *,
    opset: int = 20,
    swapped_inputs: bool = False,
    add_extra_consumer: bool = False,
):
    ide = make_node("Identity", ["x"], ["id"], name="id")
    div = make_node("Div", ["id", "sqrt2"], ["div"], name="div")
    erf = make_node("Erf", ["div"], ["erf"], name="erf")
    add_inputs = ["one", "erf"] if swapped_inputs else ["erf", "one"]
    add = make_node("Add", add_inputs, ["add"], name="add")
    mul_half_inputs = ["half", "id"] if swapped_inputs else ["id", "half"]
    mul_half = make_node("Mul", mul_half_inputs, ["mul_half"], name="mul_half")
    mul_inputs = ["mul_half", "add"] if swapped_inputs else ["add", "mul_half"]
    mul = make_node("Mul", mul_inputs, ["y"], name="mul")

    nodes = [ide, div, erf, add, mul_half, mul]
    outputs = [make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 16, 24, 24])]
    if add_extra_consumer:
        tap = make_node("Identity", ["add"], ["tap"], name="tap")
        nodes.append(tap)
        outputs.append(
            make_tensor_value_info("tap", onnx.TensorProto.FLOAT, [1, 16, 24, 24])
        )

    graph = make_graph(
        nodes,
        "gelu_graph",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 16, 24, 24])],
        outputs,
        initializer=[
            from_array(np.array(np.sqrt(2.0), dtype=np.float32), "sqrt2"),
            from_array(np.array(1.0, dtype=np.float32), "one"),
            from_array(np.array(half_value, dtype=np.float32), "half"),
        ],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[make_operatorsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_fuse_gelu():
    graph = _make_graph()
    runner1 = Evaluator(graph.model, "onnxruntime")

    graph = PassManager(["fuse_gelu"]).optimize(graph, strict=True)

    assert len(graph) == 2
    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type != "Identity":
            assert node.op_type == "Gelu"

    runner2 = Evaluator(graph.model, "onnxruntime")

    x = np.random.randn(1, 16, 24, 24).astype(np.float32)
    y1 = runner1(["y"], {"x": x})[0]
    y2 = runner2(["y"], {"x": x})[0]
    np.testing.assert_allclose(y1, y2, rtol=1e-5, atol=1e-5)


def test_skip_non_gelu_constants():
    graph = _make_graph(half_value=0.6)

    graph = PassManager(["fuse_gelu"]).optimize(graph, strict=True)

    assert len(graph) == 6
    assert graph.nodes["mul"]["pb"].op_type == "Mul"


def test_fuse_gelu_swapped_input_order():
    graph = _make_graph(swapped_inputs=True)
    runner1 = Evaluator(graph.model, "onnxruntime")

    graph = PassManager(["fuse_gelu"]).optimize(graph, strict=True)

    assert len(graph) == 2
    assert graph.nodes["mul/Gelu"]["pb"].op_type == "Gelu"

    runner2 = Evaluator(graph.model, "onnxruntime")
    x = np.random.randn(1, 16, 24, 24).astype(np.float32)
    y1 = runner1(["y"], {"x": x})[0]
    y2 = runner2(["y"], {"x": x})[0]
    np.testing.assert_allclose(y1, y2, rtol=1e-5, atol=1e-5)


def test_skip_when_intermediate_has_multiple_consumers():
    graph = _make_graph(add_extra_consumer=True)

    graph = PassManager(["fuse_gelu"]).optimize(graph, strict=True)

    assert len(graph) == 7
    assert graph.nodes["mul"]["pb"].op_type == "Mul"
    assert graph.nodes["tap"]["pb"].op_type == "Identity"


def test_skip_on_opset19():
    graph = _make_graph(opset=19)

    graph = PassManager(["fuse_gelu"]).optimize(graph, strict=True)

    assert len(graph) == 6
    assert graph.nodes["mul"]["pb"].op_type == "Mul"
