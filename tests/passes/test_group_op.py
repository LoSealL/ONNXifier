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
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.passes.fusion.group_op import _group_operators_recursively


def _make_graph():
    nodes = [
        make_node("Relu", inputs=["X"], outputs=["a0"], name="/m/act0/layer0/r0"),
        make_node("Relu", inputs=["a0"], outputs=["a1"], name="/m/act0/layer1/r1"),
        make_node("Add", inputs=["a1", "c2"], outputs=["a2"], name="/m/act0/layer2/r2"),
        make_node("Relu", inputs=["a2"], outputs=["a3"], name="/m/act1/layer3/r3"),
        make_node("Add", inputs=["a3", "c4"], outputs=["a4"], name="/m/act1/layer4/r4"),
        make_node("Relu", inputs=["a4"], outputs=["Y"], name="/m/act1/layer5/r5"),
    ]
    g = make_graph(
        nodes,
        name="hier_test",
        inputs=[
            make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1024]),
        ],
        outputs=[
            make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1024]),
        ],
        initializer=[
            from_array(np.random.rand(1024).astype(np.float32), name="c2"),
            from_array(np.random.rand(1024).astype(np.float32), name="c4"),
        ],
    )
    model = make_model(
        g, opset_imports=[ONNXIFIER_OPSET], ir_version=ONNXIFIER_IR_VERSION
    )
    onnx.checker.check_model(model)
    return model


def test_group_operators_recursively():
    model = _make_graph()
    graph = OnnxGraph(model)
    hier = {
        "act0": {
            "layer0": "/m/act0/layer0/r0",
            "layer1": "/m/act0/layer1/r1",
            "layer2": "/m/act0/layer2/r2",
        },
        "act1": {
            "layer3": "/m/act1/layer3/r3",
            "layer4": "/m/act1/layer4/r4",
            "layer5": "/m/act1/layer5/r5",
        },
    }

    node = _group_operators_recursively(graph, hier["act0"], "act0")
    assert node.name == "act0"
    assert node.input[0] == "X"
    assert node.output[0] == "a2"
    assert "act0" in graph.functions
    assert graph.functions["act0"].name == "act0"
    assert "/m/act0/layer0/r0" not in graph
    assert "/m/act0/layer1/r1" not in graph
    assert "/m/act0/layer2/r2" not in graph
    assert "/m/act1/layer3/r3" in graph
    assert "/m/act1/layer4/r4" in graph
    assert "/m/act1/layer5/r5" in graph


def test_group_depth_1():
    model = _make_graph()
    graph = OnnxGraph(model)
    graph = PassManager(
        ["group"], configs={"group": {"depth": 1, "sep": "/"}}
    ).optimize(graph, strict=True)

    # depth=1: all nodes should be grouped under "m"
    assert "m" in graph.functions
    assert graph.functions["m"].name == "m"

    # all original nodes should be removed from the main graph
    assert "/m/act0/layer0/r0" not in graph
    assert "/m/act0/layer1/r1" not in graph
    assert "/m/act0/layer2/r2" not in graph
    assert "/m/act1/layer3/r3" not in graph
    assert "/m/act1/layer4/r4" not in graph
    assert "/m/act1/layer5/r5" not in graph

    # the function call node should exist
    assert "m" in graph
    m_node = graph.nodes["m"]["pb"]
    assert m_node.op_type == "m"
    assert m_node.input[0] == "X"
    assert m_node.output[0] == "Y"


def test_group_depth_2():
    model = _make_graph()
    graph = OnnxGraph(model)
    graph = PassManager(
        ["group"], configs={"group": {"depth": 2, "sep": "/"}}
    ).optimize(graph, strict=True)

    # depth=2: nodes should be grouped under "m" -> "act0"/"act1"
    assert "m" in graph.functions
    assert "act0" in graph.functions
    assert "act1" in graph.functions

    # all original nodes should be removed from the main graph
    assert "/m/act0/layer0/r0" not in graph
    assert "/m/act0/layer1/r1" not in graph
    assert "/m/act0/layer2/r2" not in graph
    assert "/m/act1/layer3/r3" not in graph
    assert "/m/act1/layer4/r4" not in graph
    assert "/m/act1/layer5/r5" not in graph

    # the top-level function call node should exist
    assert "m" in graph
    m_node = graph.nodes["m"]["pb"]
    assert m_node.op_type == "m"
    assert m_node.input[0] == "X"
    assert m_node.output[0] == "Y"


def test_group_depth_3():
    # depth=3: nodes should be grouped under
    # "m" -> "act0"/"act1" -> "layer0"/"layer1"/...
    model = _make_graph()
    graph = OnnxGraph(model)
    graph = PassManager(
        ["group"], configs={"group": {"depth": 3, "sep": "/"}}
    ).optimize(graph, strict=True)

    assert "m" in graph.functions
    assert "act0" in graph.functions
    assert "act1" in graph.functions
    assert "layer0" in graph.functions
    assert "layer1" in graph.functions
    assert "layer2" in graph.functions
    assert "layer3" in graph.functions
    assert "layer4" in graph.functions
    assert "layer5" in graph.functions

    # all original nodes should be removed from the main graph
    assert "/m/act0/layer0/r0" not in graph
    assert "/m/act0/layer1/r1" not in graph
    assert "/m/act0/layer2/r2" not in graph
    assert "/m/act1/layer3/r3" not in graph
    assert "/m/act1/layer4/r4" not in graph
    assert "/m/act1/layer5/r5" not in graph

    # the top-level function call node should exist
    assert "m" in graph
    m_node = graph.nodes["m"]["pb"]
    assert m_node.op_type == "m"
    assert m_node.input[0] == "X"
    assert m_node.output[0] == "Y"
