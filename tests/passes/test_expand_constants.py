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
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier import ONNXIFIER_OPSET, OnnxGraph
from onnxifier.passes.utils import expand_constants


def _make_graph_with_constant():
    """Create a graph with constant node feeding into an Add node."""
    const = make_node(
        "Constant",
        [],
        ["const_val"],
        name="const_node",
        value=onnx.numpy_helper.from_array(np.ones([10], dtype=np.float32)),
    )
    add = make_node("Add", ["x", "const_val"], ["y"], name="add_node")
    graph = make_graph(
        [const, add],
        "test_graph",
        [make_value_info("x", make_tensor_type_proto(1, [10]))],
        [make_value_info("y", make_tensor_type_proto(1, [10]))],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def _make_graph_with_initializer():
    """Create a graph with initializer feeding into an Add node."""
    add = make_node("Add", ["x", "init_val"], ["y"], name="add_node")
    graph = make_graph(
        [add],
        "test_graph",
        [make_value_info("x", make_tensor_type_proto(1, [10]))],
        [make_value_info("y", make_tensor_type_proto(1, [10]))],
        [onnx.numpy_helper.from_array(np.ones([10], dtype=np.float32), "init_val")],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def _make_graph_with_chained_constants():
    """Create a graph with chained constant nodes."""
    const1 = make_node(
        "Constant",
        [],
        ["c1"],
        name="const1",
        value=onnx.numpy_helper.from_array(np.ones([10], dtype=np.float32)),
    )
    const2 = make_node(
        "Constant",
        [],
        ["c2"],
        name="const2",
        value=onnx.numpy_helper.from_array(np.ones([10], dtype=np.float32)),
    )
    add1 = make_node("Add", ["c1", "c2"], ["c3"], name="add1")
    add2 = make_node("Add", ["c3", "x"], ["y"], name="add2")
    graph = make_graph(
        [const1, const2, add1, add2],
        "test_graph",
        [make_value_info("x", make_tensor_type_proto(1, [10]))],
        [make_value_info("y", make_tensor_type_proto(1, [10]))],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def _make_graph_with_dynamic_constant_path():
    """Create a graph where constant path has a dynamic input."""
    const1 = make_node(
        "Constant",
        [],
        ["c1"],
        name="const1",
        value=onnx.numpy_helper.from_array(np.ones([10], dtype=np.float32)),
    )
    mul = make_node("Mul", ["x", "c1"], ["m"], name="mul")
    add = make_node("Add", ["m", "z"], ["y"], name="add")
    graph = make_graph(
        [const1, mul, add],
        "test_graph",
        [
            make_value_info("x", make_tensor_type_proto(1, [10])),
            make_value_info("z", make_tensor_type_proto(1, [10])),
        ],
        [make_value_info("y", make_tensor_type_proto(1, [10]))],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def test_expand_constants_basic():
    """Test expand_constants includes upstream constant node."""
    model = _make_graph_with_constant()
    graph = OnnxGraph(model)

    add_node = graph.nodes["add_node"]["pb"]
    expanded = expand_constants(graph, [add_node])

    # Should include both the Add node and the upstream Constant node
    assert len(expanded) == 2
    node_names = [n.name for n in expanded]
    assert "add_node" in node_names
    assert "const_node" in node_names


def test_expand_constants_with_initializer():
    """Test expand_constants skips initializers."""
    model = _make_graph_with_initializer()
    graph = OnnxGraph(model)

    add_node = graph.nodes["add_node"]["pb"]
    expanded = expand_constants(graph, [add_node])

    # Should only include the Add node (initializer is not expanded)
    assert len(expanded) == 1
    assert expanded[0].name == "add_node"


def test_expand_constants_with_input():
    """Test expand_constants skips graph inputs."""
    model = _make_graph_with_constant()
    graph = OnnxGraph(model)

    add_node = graph.nodes["add_node"]["pb"]
    expanded = expand_constants(graph, [add_node])

    # Input 'x' should not be expanded (it's a graph input)
    # Only the Constant node feeding into 'const_val' should be expanded
    assert len(expanded) == 2
    node_names = [n.name for n in expanded]
    assert "add_node" in node_names
    assert "const_node" in node_names


def test_expand_constants_chained():
    """Test expand_constants includes chained constant nodes."""
    model = _make_graph_with_chained_constants()
    graph = OnnxGraph(model)

    add2_node = graph.nodes["add2"]["pb"]
    expanded = expand_constants(graph, [add2_node])

    # Should include add2, add1, const1, const2 (all upstream constants)
    assert len(expanded) == 4
    node_names = [n.name for n in expanded]
    assert "add2" in node_names
    assert "add1" in node_names
    assert "const1" in node_names
    assert "const2" in node_names


def test_expand_constants_dynamic_path():
    """Test expand_constants handles nodes with dynamic inputs."""
    model = _make_graph_with_dynamic_constant_path()
    graph = OnnxGraph(model)

    add_node = graph.nodes["add"]["pb"]
    expanded = expand_constants(graph, [add_node])

    # The Add node has input 'm' from Mul, and 'z' from graph input
    # Mul has input 'x' (graph input) and 'c1' (constant)
    # Since 'm' comes from a node with input 'x', it should not expand
    # unless it can be evaluated to constant (which depends on runtime)
    # In this case, mul has dynamic input 'x', so it shouldn't be expanded
    assert len(expanded) == 1
    assert expanded[0].name == "add"


def test_expand_constants_multiple_nodes():
    """Test expand_constants with multiple input nodes."""
    model = _make_graph_with_chained_constants()
    graph = OnnxGraph(model)

    add1_node = graph.nodes["add1"]["pb"]
    add2_node = graph.nodes["add2"]["pb"]
    expanded = expand_constants(graph, [add1_node, add2_node])

    # Should include all nodes and their upstream constants
    assert len(expanded) == 4
    node_names = [n.name for n in expanded]
    assert "add1" in node_names
    assert "add2" in node_names
    assert "const1" in node_names
    assert "const2" in node_names


def test_expand_constants_no_duplicates():
    """Test expand_constants doesn't add duplicate nodes."""
    model = _make_graph_with_constant()
    graph = OnnxGraph(model)

    add_node = graph.nodes["add_node"]["pb"]
    const_node = graph.nodes["const_node"]["pb"]

    # Pass both nodes, const_node is upstream of add_node
    expanded = expand_constants(graph, [add_node, const_node])

    # Should not have duplicates
    node_names = [n.name for n in expanded]
    assert len(node_names) == len(set(node_names))
    assert "add_node" in node_names
    assert "const_node" in node_names
