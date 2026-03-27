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

import networkx as nx
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

    [node, *_] = _group_operators_recursively(graph, hier["act0"], "act0")
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


def test_group_connectivity_split():
    """Test that disconnected nodes with the same prefix are split into separate
    groups.
    """
    # Create a graph with two disconnected sub-graphs that have the same prefix
    # Graph 1: X -> Relu (name=/branch/relu0) -> y1
    # Graph 2: Z -> Relu (name=/branch/relu1) -> y2
    nodes = [
        make_node("Relu", inputs=["X"], outputs=["y1"], name="/branch/relu0"),
        make_node("Relu", inputs=["Z"], outputs=["y2"], name="/branch/relu1"),
    ]
    g = make_graph(
        nodes,
        name="connectivity_test",
        inputs=[
            make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1024]),
            make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [1024]),
        ],
        outputs=[
            make_tensor_value_info("y1", onnx.TensorProto.FLOAT, [1024]),
            make_tensor_value_info("y2", onnx.TensorProto.FLOAT, [1024]),
        ],
    )
    model = make_model(
        g, opset_imports=[ONNXIFIER_OPSET], ir_version=ONNXIFIER_IR_VERSION
    )
    onnx.checker.check_model(model)

    graph = OnnxGraph(model)
    graph = PassManager(
        ["group"], configs={"group": {"depth": 2, "sep": "/"}}
    ).optimize(graph, strict=True)

    # The two disconnected relu nodes should be split into separate functions
    # with names "branch_0" and "branch_1" (or similar)
    assert "branch_0" in graph.functions
    assert "branch_1" in graph.functions

    # Check that the original nodes are removed
    assert "/branch/relu0" not in graph
    assert "/branch/relu1" not in graph

    # Check that both function call nodes exist in the graph
    assert "branch_0" in graph
    assert "branch_1" in graph


def test_group_no_cycle_through_split():
    """Test that splitting disconnected /a nodes prevents a cycle with /b.

    Graph topology:
        X -> (/a/0) -> (/b/1) -> (/a/1) -> Y

    At depth=1, /a/0 and /a/1 share prefix "a", but they are NOT directly
    connected — /b/1 sits between them.  If the pass naively grouped /a/0 and
    /a/1 together, the resulting DAG would be:

        a <-> b  (cycle)

    The correct behaviour is to split "a" into two separate functions:
        a_0  (contains /a/0)
        a_1  (contains /a/1)

    so the final DAG is cycle-free:
        a_0 -> b -> a_1
    """
    nodes = [
        make_node("Relu", inputs=["X"], outputs=["t0"], name="/a/0"),
        make_node("Relu", inputs=["t0"], outputs=["t1"], name="/b/1"),
        make_node("Relu", inputs=["t1"], outputs=["Y"], name="/a/1"),
    ]
    g = make_graph(
        nodes,
        name="cycle_test",
        inputs=[make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1024])],
        outputs=[make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1024])],
    )
    model = make_model(
        g, opset_imports=[ONNXIFIER_OPSET], ir_version=ONNXIFIER_IR_VERSION
    )
    onnx.checker.check_model(model)

    graph = OnnxGraph(model)
    graph = PassManager(
        ["group"], configs={"group": {"depth": 1, "sep": "/"}}
    ).optimize(graph, strict=True)

    # The resulting function-call graph must be acyclic
    assert nx.is_directed_acyclic_graph(graph)
    # /a/0 and /a/1 must be in separate groups because merging them would
    # create a cycle  a <-> b.
    assert "a_0" in graph.functions
    assert "a_1" in graph.functions
    assert "b" in graph.functions

    # original nodes are gone
    assert "/a/0" not in graph
    assert "/b/1" not in graph
    assert "/a/1" not in graph


def test_group_no_cycle_through_split_nested():
    """Test cycle prevention with nested hierarchy (depth=2) and 5 nodes.

    Graph topology (linear chain):
        X -> (/outer/a/n0) -> (/outer/b/n1) -> (/outer/a/n2)
          -> (/outer/b/n3) -> (/outer/a/n4) -> Y

    At depth=2 the inner grouping is:
      "a" wants to merge n0, n2, n4 — but they are NOT directly connected
          (b-nodes sit between them), so they must be split into a_0/a_1/a_2.
      "b" wants to merge n1, n3 — similarly disconnected → b_0/b_1.

    Naively merging either set would create a cycle in the grouped DAG.
    After correct connectivity splitting the result must be acyclic:
        a_0 -> b_0 -> a_1 -> b_1 -> a_2
    """
    nodes = [
        make_node("Relu", inputs=["X"], outputs=["t0"], name="/outer/a/n0"),
        make_node("Relu", inputs=["t0"], outputs=["t1"], name="/outer/b/n1"),
        make_node("Relu", inputs=["t1"], outputs=["t2"], name="/outer/a/n2"),
        make_node("Relu", inputs=["t2"], outputs=["t3"], name="/outer/b/n3"),
        make_node("Relu", inputs=["t3"], outputs=["Y"], name="/outer/a/n4"),
    ]
    g = make_graph(
        nodes,
        name="nested_cycle_test",
        inputs=[make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1024])],
        outputs=[make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1024])],
    )
    model = make_model(
        g, opset_imports=[ONNXIFIER_OPSET], ir_version=ONNXIFIER_IR_VERSION
    )
    onnx.checker.check_model(model)

    graph = OnnxGraph(model)
    graph = PassManager(
        ["group"], configs={"group": {"depth": 2, "sep": "/"}}
    ).optimize(graph, strict=True)

    # Most important: the grouped graph must remain acyclic
    assert nx.is_directed_acyclic_graph(graph)

    # "a" splits into 3 disconnected components, "b" into 2
    assert "a_0" in graph.functions
    assert "a_1" in graph.functions
    assert "a_2" in graph.functions
    assert "b_0" in graph.functions
    assert "b_1" in graph.functions

    # All original nodes are removed from the main graph
    assert "/outer/a/n0" not in graph
    assert "/outer/b/n1" not in graph
    assert "/outer/a/n2" not in graph
    assert "/outer/b/n3" not in graph
    assert "/outer/a/n4" not in graph


def test_group_cycle_merge_flat():
    nodes = [
        make_node("Relu", inputs=["X"], outputs=["t0"], name="/a/n0"),
        make_node("Relu", inputs=["t0"], outputs=["t1"], name="/b/n0"),
        make_node("Relu", inputs=["t0"], outputs=["t2"], name="/b/n1"),
        make_node(
            "Concat", inputs=["t0", "t1", "t2"], outputs=["Y"], name="/a/n1", axis=0
        ),
    ]
    g = make_graph(
        nodes,
        name="cycle_merge_flat",
        inputs=[make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1024])],
        outputs=[make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1024])],
    )
    model = make_model(
        g, opset_imports=[ONNXIFIER_OPSET], ir_version=ONNXIFIER_IR_VERSION
    )
    onnx.checker.check_model(model)

    graph = OnnxGraph(model)
    graph = PassManager(
        ["group"], configs={"group": {"depth": 1, "sep": "/"}}
    ).optimize(graph, strict=True)

    assert nx.is_directed_acyclic_graph(graph)
    assert "a" in graph.functions
    assert "b" not in graph.functions
