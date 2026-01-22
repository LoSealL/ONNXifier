"""
Copyright (C) 2025 The ONNXIFIER Authors.

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

from onnxifier import OnnxGraph, PassManager


def test_constant_to_constantofshape_fp32():
    """Test conversion of fp32 constant to ConstantOfShape with threshold."""
    # Create a constant node with enough elements to exceed default threshold (16)
    constant_value = np.full((4, 5), 1.5, dtype=np.float32)  # 20 elements > 16
    constant_node = make_node(
        "Constant",
        inputs=[],
        outputs=["const_output"],
        value=onnx.numpy_helper.from_array(constant_value, "const_value"),
    )

    graph = make_graph(
        [constant_node],
        "test_graph",
        [],
        [make_tensor_value_info("const_output", onnx.TensorProto.FLOAT, [4, 5])],
    )
    model = make_model(graph)
    onnx.checker.check_model(model, True)

    # Apply the pass
    onnx_graph = OnnxGraph(model)
    pm = PassManager(["constant_to_constantofshape"])
    optimized_graph = pm.optimize(onnx_graph, strict=True)

    # Verify the result
    nodes = [optimized_graph.nodes[node]["pb"] for node in optimized_graph.nodes]

    # Should have two nodes: one shape constant and one ConstantOfShape
    assert len(nodes) == 2

    shape_nodes = [n for n in nodes if n.op_type == "Constant"]
    constantofshape_nodes = [n for n in nodes if n.op_type == "ConstantOfShape"]

    assert len(shape_nodes) == 1
    assert len(constantofshape_nodes) == 1

    # Verify the ConstantOfShape node has correct value
    cos_node = constantofshape_nodes[0]
    value_attr = next(attr for attr in cos_node.attribute if attr.name == "value")
    value_array = onnx.numpy_helper.to_array(value_attr.t)
    assert value_array.dtype == np.float32
    assert value_array[0] == 1.5


def test_constant_to_constantofshape_fp16():
    """Test conversion of fp16 constant to ConstantOfShape with threshold."""
    # Create a constant node with enough elements to exceed default threshold (16)
    constant_value = np.full((3, 6), 2.5, dtype=np.float16)  # 18 elements > 16
    constant_node = make_node(
        "Constant",
        inputs=[],
        outputs=["const_output"],
        value=onnx.numpy_helper.from_array(constant_value, "const_value"),
    )

    graph = make_graph(
        [constant_node],
        "test_graph",
        [],
        [make_tensor_value_info("const_output", onnx.TensorProto.FLOAT16, [3, 6])],
    )
    model = make_model(graph)
    onnx.checker.check_model(model, True)

    # Apply the pass
    onnx_graph = OnnxGraph(model)
    pm = PassManager(["constant_to_constantofshape"])
    optimized_graph = pm.optimize(onnx_graph, strict=True)

    # Verify the result
    nodes = [optimized_graph.nodes[node]["pb"] for node in optimized_graph.nodes]

    # Should have two nodes: one shape constant and one ConstantOfShape
    assert len(nodes) == 2

    constantofshape_nodes = [n for n in nodes if n.op_type == "ConstantOfShape"]
    assert len(constantofshape_nodes) == 1

    # Verify the ConstantOfShape node has correct value
    cos_node = constantofshape_nodes[0]
    value_attr = next(attr for attr in cos_node.attribute if attr.name == "value")
    value_array = onnx.numpy_helper.to_array(value_attr.t)
    assert value_array.dtype == np.float16
    assert value_array[0] == np.float16(2.5)


def test_constant_to_constantofshape_int32():
    """Test conversion of int32 constant to ConstantOfShape with threshold."""
    # Create a constant node with enough elements to exceed default threshold (16)
    constant_value = np.full((4, 5), 42, dtype=np.int32)  # 20 elements > 16
    constant_node = make_node(
        "Constant",
        inputs=[],
        outputs=["const_output"],
        value=onnx.numpy_helper.from_array(constant_value, "const_value"),
    )

    graph = make_graph(
        [constant_node],
        "test_graph",
        [],
        [make_tensor_value_info("const_output", onnx.TensorProto.INT32, [4, 5])],
    )
    model = make_model(graph)
    onnx.checker.check_model(model, True)

    # Apply the pass
    onnx_graph = OnnxGraph(model)
    pm = PassManager(["constant_to_constantofshape"])
    optimized_graph = pm.optimize(onnx_graph, strict=True)

    # Verify the result
    nodes = [optimized_graph.nodes[node]["pb"] for node in optimized_graph.nodes]

    # Should have two nodes: one shape constant and one ConstantOfShape
    assert len(nodes) == 2

    constantofshape_nodes = [n for n in nodes if n.op_type == "ConstantOfShape"]
    assert len(constantofshape_nodes) == 1

    # Verify the ConstantOfShape node has correct value
    cos_node = constantofshape_nodes[0]
    value_attr = next(attr for attr in cos_node.attribute if attr.name == "value")
    value_array = onnx.numpy_helper.to_array(value_attr.t)
    assert value_array.dtype == np.int32
    assert value_array[0] == 42


def test_constant_to_constantofshape_int64():
    """Test conversion of int64 constant to ConstantOfShape with threshold."""
    # Create a constant node with enough elements to exceed default threshold (16)
    constant_value = np.full((3, 6), 123456789, dtype=np.int64)  # 18 elements > 16
    constant_node = make_node(
        "Constant",
        inputs=[],
        outputs=["const_output"],
        value=onnx.numpy_helper.from_array(constant_value, "const_value"),
    )

    graph = make_graph(
        [constant_node],
        "test_graph",
        [],
        [make_tensor_value_info("const_output", onnx.TensorProto.INT64, [3, 6])],
    )
    model = make_model(graph)
    onnx.checker.check_model(model, True)

    # Apply the pass
    onnx_graph = OnnxGraph(model)
    pm = PassManager(["constant_to_constantofshape"])
    optimized_graph = pm.optimize(onnx_graph, strict=True)

    # Verify the result
    nodes = [optimized_graph.nodes[node]["pb"] for node in optimized_graph.nodes]

    # Should have two nodes: one shape constant and one ConstantOfShape
    assert len(nodes) == 2

    constantofshape_nodes = [n for n in nodes if n.op_type == "ConstantOfShape"]
    assert len(constantofshape_nodes) == 1

    # Verify the ConstantOfShape node has correct value
    cos_node = constantofshape_nodes[0]
    value_attr = next(attr for attr in cos_node.attribute if attr.name == "value")
    value_array = onnx.numpy_helper.to_array(value_attr.t)
    assert value_array.dtype == np.int64
    assert value_array[0] == 123456789


def test_constant_to_constantofshape_skip_under_threshold():
    """Test that constants below threshold are not converted."""
    # Create a constant node with elements below threshold
    # 4 elements < 16
    constant_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    constant_node = make_node(
        "Constant",
        inputs=[],
        outputs=["const_output"],
        value=onnx.numpy_helper.from_array(constant_value, "const_value"),
    )

    graph = make_graph(
        [constant_node],
        "test_graph",
        [],
        [make_tensor_value_info("const_output", onnx.TensorProto.FLOAT, [2, 2])],
    )
    model = make_model(graph)
    onnx.checker.check_model(model, True)

    # Apply the pass
    onnx_graph = OnnxGraph(model)
    pm = PassManager(["constant_to_constantofshape"])
    optimized_graph = pm.optimize(onnx_graph, strict=True)

    # Verify the result - should remain unchanged
    nodes = [optimized_graph.nodes[node]["pb"] for node in optimized_graph.nodes]
    assert len(nodes) == 1
    assert nodes[0].op_type == "Constant"


def test_constant_to_constantofshape_scalar():
    """Test that scalar constants below threshold are not converted."""
    # Create a scalar constant node (1 element < 16)
    constant_value = np.array(3.14, dtype=np.float32)
    constant_node = make_node(
        "Constant",
        inputs=[],
        outputs=["const_output"],
        value=onnx.numpy_helper.from_array(constant_value, "const_value"),
    )

    graph = make_graph(
        [constant_node],
        "test_graph",
        [],
        [make_tensor_value_info("const_output", onnx.TensorProto.FLOAT, [])],
    )
    model = make_model(graph)
    onnx.checker.check_model(model, True)

    # Apply the pass
    onnx_graph = OnnxGraph(model)
    pm = PassManager(["constant_to_constantofshape"])
    optimized_graph = pm.optimize(onnx_graph, strict=True)

    # Verify the result - should remain unchanged (below threshold)
    nodes = [optimized_graph.nodes[node]["pb"] for node in optimized_graph.nodes]
    assert len(nodes) == 1
    assert nodes[0].op_type == "Constant"


def test_constant_to_constantofshape_with_custom_threshold():
    """Test conversion with custom threshold."""
    # Create a constant node with 10 elements
    constant_value = np.full((2, 5), 1.5, dtype=np.float32)  # 10 elements
    constant_node = make_node(
        "Constant",
        inputs=[],
        outputs=["const_output"],
        value=onnx.numpy_helper.from_array(constant_value, "const_value"),
    )

    graph = make_graph(
        [constant_node],
        "test_graph",
        [],
        [make_tensor_value_info("const_output", onnx.TensorProto.FLOAT, [2, 5])],
    )
    model = make_model(graph)
    onnx.checker.check_model(model, True)

    # Apply the pass with threshold=5 (should convert)
    onnx_graph = OnnxGraph(model)
    from onnxifier.passes.swap.constant_to_constantofshape import (
        ConstantToConstantOfShapeRewriter,
    )

    # Create rewriter with custom threshold
    rewriter = ConstantToConstantOfShapeRewriter(threshold=5)
    onnx_graph = rewriter.match_and_rewrite(onnx_graph)

    # Verify the result - should be converted
    nodes = [onnx_graph.nodes[node]["pb"] for node in onnx_graph.nodes]
    assert len(nodes) == 2

    constantofshape_nodes = [n for n in nodes if n.op_type == "ConstantOfShape"]
    assert len(constantofshape_nodes) == 1

    # Verify the ConstantOfShape node has correct value (first element)
    cos_node = constantofshape_nodes[0]
    value_attr = next(attr for attr in cos_node.attribute if attr.name == "value")
    value_array = onnx.numpy_helper.to_array(value_attr.t)
    assert value_array.dtype == np.float32
    assert value_array[0] == 1.5


def test_constant_to_constantofshape_takes_first_element():
    """Test that conversion takes the first element regardless of uniformity."""
    # Create a constant node with non-uniform values above threshold
    constant_value = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
        ],
        dtype=np.float32,
    )  # 20 elements > 16, non-uniform
    constant_node = make_node(
        "Constant",
        inputs=[],
        outputs=["const_output"],
        value=onnx.numpy_helper.from_array(constant_value, "const_value"),
    )

    graph = make_graph(
        [constant_node],
        "test_graph",
        [],
        [make_tensor_value_info("const_output", onnx.TensorProto.FLOAT, [4, 5])],
    )
    model = make_model(graph)
    onnx.checker.check_model(model, True)

    # Apply the pass
    onnx_graph = OnnxGraph(model)
    pm = PassManager(["constant_to_constantofshape"])
    optimized_graph = pm.optimize(onnx_graph, strict=True)

    # Verify the result - should be converted
    nodes = [optimized_graph.nodes[node]["pb"] for node in optimized_graph.nodes]
    assert len(nodes) == 2

    constantofshape_nodes = [n for n in nodes if n.op_type == "ConstantOfShape"]
    assert len(constantofshape_nodes) == 1

    # Verify the ConstantOfShape node has the first element value (1.0)
    cos_node = constantofshape_nodes[0]
    value_attr = next(attr for attr in cos_node.attribute if attr.name == "value")
    value_array = onnx.numpy_helper.to_array(value_attr.t)
    assert value_array.dtype == np.float32
    assert value_array[0] == 1.0  # First element of the original array
