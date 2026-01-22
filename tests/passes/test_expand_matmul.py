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
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_type_proto,
    make_value_info,
)
from onnx.reference import ReferenceEvaluator

from onnxifier import PassManager
from onnxifier.graph import OnnxGraph


def _build_matmul_graph():
    """Build a simple MatMul graph: input (2, 3) @ weight (3, 5) -> output (2, 5)"""
    matmul = make_node("MatMul", ["x", "w"], ["y"], "matmul")
    graph = make_graph(
        [matmul],
        "matmul_graph",
        [make_value_info("x", make_tensor_type_proto(1, [2, 3]))],
        [make_value_info("y", make_tensor_type_proto(1, [2, 5]))],
        [make_tensor("w", 1, [3, 5], np.random.rand(3, 5).astype("float32"))],
    )
    return make_model(graph)


def _build_batch_matmul_graph():
    """Build a batch MatMul graph: input (2, 3, 5) @ weight (2, 5, 7)
    -> output (2, 3, 7)
    """
    matmul = make_node("MatMul", ["x", "w"], ["y"], "batch_matmul")
    graph = make_graph(
        [matmul],
        "batch_matmul_graph",
        [make_value_info("x", make_tensor_type_proto(1, [2, 3, 5]))],
        [make_value_info("y", make_tensor_type_proto(1, [2, 3, 7]))],
        [make_tensor("w", 1, [2, 5, 7], np.random.rand(2, 5, 7).astype("float32"))],
    )
    return make_model(graph)


def test_expand_matmul_basic():
    """Test basic 2D MatMul expansion using PassManager and ReferenceEvaluator"""
    # Build original model
    model = _build_matmul_graph()
    runner1 = ReferenceEvaluator(model)

    # Create test input
    x = np.random.uniform(0, 1, size=[2, 3]).astype(np.float32)
    y1_result = runner1.run(None, {"x": x})
    y1 = y1_result[0] if isinstance(y1_result, list) else list(y1_result.values())[0]

    # Apply expansion pass
    graph = OnnxGraph(model)
    pm = PassManager(["expand_matmul"])
    graph = pm.optimize(graph, strict=True)

    # Test with same input shape (padding is handled internally)
    runner2 = ReferenceEvaluator(graph.model)

    # Use the same input - no need to expand manually
    y2_result = runner2.run(None, {"x": x})
    y2 = y2_result[0] if isinstance(y2_result, list) else list(y2_result.values())[0]

    # Results should be identical
    assert np.allclose(
        y1, y2, rtol=1e-5, atol=1e-6
    ), f"Basic MatMul expansion failed. Max diff: {np.max(np.abs(y1 - y2))}"


def test_expand_matmul_batch():
    """Test batch MatMul expansion - batch dimensions should not be expanded"""
    # Build original batch model
    model = _build_batch_matmul_graph()
    runner1 = ReferenceEvaluator(model)

    # Create test input
    x = np.random.uniform(0, 1, size=[2, 3, 5]).astype(np.float32)
    y1_result = runner1.run(None, {"x": x})
    y1 = y1_result[0] if isinstance(y1_result, list) else list(y1_result.values())[0]

    # Apply expansion pass
    graph = OnnxGraph(model)
    pm = PassManager(["expand_matmul"])
    graph = pm.optimize(graph, strict=True)

    # Test with same input shape (padding is handled internally)
    runner2 = ReferenceEvaluator(graph.model)

    # Use the same input - no need to expand manually
    y2_result = runner2.run(None, {"x": x})
    y2 = y2_result[0] if isinstance(y2_result, list) else list(y2_result.values())[0]

    # Results should be identical
    assert np.allclose(
        y1, y2, rtol=1e-5, atol=1e-6
    ), f"Batch MatMul expansion failed. Max diff: {np.max(np.abs(y1 - y2))}"


def test_expand_matmul_no_expansion_needed():
    """Test that already aligned dimensions don't get expanded"""
    # Create a MatMul with dimensions already aligned to factor=2
    matmul = make_node("MatMul", ["x", "w"], ["y"], "matmul")
    graph = make_graph(
        [matmul],
        "aligned_matmul_graph",
        [make_value_info("x", make_tensor_type_proto(1, [4, 6]))],  # Already aligned
        [make_value_info("y", make_tensor_type_proto(1, [4, 6]))],
        [
            make_tensor(
                "w",
                1,
                [6, 6],
                np.random.rand(6, 6).astype("float32"),  # Already aligned
            )
        ],
    )
    model = make_model(graph)

    # Apply expansion pass
    original_graph = OnnxGraph(model)
    pm = PassManager(["expand_matmul"])
    optimized_graph = pm.optimize(original_graph, strict=True)

    # Graph should remain unchanged (same number of nodes)
    assert len(optimized_graph.nodes) == len(
        original_graph.nodes
    ), "Graph was modified when no expansion was needed"


def test_expand_matmul_numerical_accuracy():
    """Comprehensive numerical accuracy test for different shapes"""
    test_cases = [
        ([3, 5], [5, 7], "2D MatMul"),
        ([2, 3, 5], [2, 5, 7], "3D Batch MatMul"),
        ([5, 13], [13, 11], "2D with odd dimensions"),
    ]

    for input_shape, weight_shape, description in test_cases:
        print(f"\nTesting {description}: {input_shape} @ {weight_shape}")

        # Build model for this test case
        matmul = make_node("MatMul", ["x", "w"], ["y"], "matmul")
        output_shape = input_shape[:-1] + [weight_shape[-1]]

        graph = make_graph(
            [matmul],
            f"test_{description.replace(' ', '_').lower()}",
            [make_value_info("x", make_tensor_type_proto(1, input_shape))],
            [make_value_info("y", make_tensor_type_proto(1, output_shape))],
            [
                make_tensor(
                    "w",
                    1,
                    weight_shape,
                    np.random.rand(*weight_shape).astype("float32"),
                )
            ],
        )
        model = make_model(graph)

        # Run original model
        runner1 = ReferenceEvaluator(model)
        x = np.random.uniform(0, 1, size=input_shape).astype(np.float32)
        y1_result = runner1.run(None, {"x": x})
        if isinstance(y1_result, list):
            y1 = y1_result[0]
        else:
            y1 = list(y1_result.values())[0]

        # Apply expansion pass
        graph_obj = OnnxGraph(model)
        pm = PassManager(["expand_matmul"])
        optimized_graph = pm.optimize(graph_obj, strict=True)

        # Run optimized model with same input (padding handled internally)
        runner2 = ReferenceEvaluator(optimized_graph.model)
        y2_result = runner2.run(None, {"x": x})
        if isinstance(y2_result, list):
            y2 = y2_result[0]
        else:
            y2 = list(y2_result.values())[0]

        # Check numerical accuracy
        max_diff = np.max(np.abs(y1 - y2))
        mean_diff = np.mean(np.abs(y1 - y2))

        assert np.allclose(
            y1, y2, rtol=1e-5, atol=1e-6
        ), f"{description} failed. Max diff: {max_diff}, Mean diff: {mean_diff}"


def test_expand_matmul_factor_2():
    """Test with factor=2 to verify the new default"""
    # Test case specifically designed for factor=2
    input_shape = [3, 5]  # Should expand to [4, 6]
    weight_shape = [5, 7]  # Should expand to [6, 8]

    matmul = make_node("MatMul", ["x", "w"], ["y"], "matmul")
    output_shape = input_shape[:-1] + [weight_shape[-1]]

    graph = make_graph(
        [matmul],
        "test_factor_2",
        [make_value_info("x", make_tensor_type_proto(1, input_shape))],
        [make_value_info("y", make_tensor_type_proto(1, output_shape))],
        [
            make_tensor(
                "w", 1, weight_shape, np.random.rand(*weight_shape).astype("float32")
            )
        ],
    )
    model = make_model(graph)

    # Run original model
    runner1 = ReferenceEvaluator(model)
    x = np.random.uniform(0, 1, size=input_shape).astype(np.float32)
    y1_result = runner1.run(None, {"x": x})
    y1 = y1_result[0] if isinstance(y1_result, list) else list(y1_result.values())[0]

    # Apply expansion pass with default factor=2
    graph_obj = OnnxGraph(model)
    pm = PassManager(["expand_matmul"])
    optimized_graph = pm.optimize(graph_obj, strict=True)

    # Run optimized model
    runner2 = ReferenceEvaluator(optimized_graph.model)
    y2_result = runner2.run(None, {"x": x})
    if isinstance(y2_result, list):
        y2 = y2_result[0]
    else:
        y2 = list(y2_result.values())[0]

    # Check numerical accuracy
    max_diff = np.max(np.abs(y1 - y2))
    mean_diff = np.mean(np.abs(y1 - y2))

    assert np.allclose(
        y1, y2, rtol=1e-5, atol=1e-6
    ), f"Factor=2 test failed. Max diff: {max_diff}, Mean diff: {mean_diff}"

    # Verify that expansion actually happened by checking node count
    original_node_count = len(model.graph.node)
    optimized_node_count = len(optimized_graph.model.graph.node)
    assert (
        optimized_node_count > original_node_count
    ), "Expected more nodes after expansion (Pad, expanded MatMul, Slice)"


if __name__ == "__main__":
    test_expand_matmul_basic()
    test_expand_matmul_batch()
    test_expand_matmul_no_expansion_needed()
    test_expand_matmul_numerical_accuracy()
    test_expand_matmul_factor_2()
