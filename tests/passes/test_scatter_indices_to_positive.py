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
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_type_proto,
    make_value_info,
)
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, PassManager
from onnxifier.graph import OnnxGraph
from onnxifier.passes.utils import evaluate_on_node


def _make_scatter_nd_model(indices_data, data_shape):
    """Create a ScatterND model with given indices and data shape."""
    # Create input data tensor info
    data_info = make_value_info(
        "data", make_tensor_type_proto(TensorProto.FLOAT, data_shape)
    )

    # For ScatterND, updates shape is
    # indices.shape[:-1] + data_shape[indices.shape[-1]:]
    updates_shape = list(indices_data.shape[:-1]) + data_shape[indices_data.shape[-1] :]
    updates_info = make_value_info(
        "updates", make_tensor_type_proto(TensorProto.FLOAT, updates_shape)
    )

    # Create output tensor info
    output_info = make_value_info(
        "output", make_tensor_type_proto(TensorProto.FLOAT, data_shape)
    )

    # Create indices initializer
    indices_tensor = from_array(indices_data, "indices")

    # Create scatter node
    scatter_node = make_node(
        "ScatterND", ["data", "indices", "updates"], ["output"], name="scatter_nd"
    )

    # Create graph
    graph = make_graph(
        [scatter_node],
        "scatter_nd_graph",
        [data_info, updates_info],
        [output_info],
        [indices_tensor],
    )

    # Create model
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    return model


def test_no_negative_indices():
    """Test case where indices are already positive."""
    # Create indices with no negative values
    # For data_shape [2, 3, 4], indices should have shape [..., 3]
    # where 3 is the rank of data
    indices = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    data_shape = [2, 3, 4]

    model = _make_scatter_nd_model(indices, data_shape)
    graph = OnnxGraph(model)

    # Apply pass
    pm = PassManager(["scatter_nd_indices_to_positive"])
    graph = pm.optimize(graph, strict=True)

    assert "scatter_nd/indices_pos" not in graph


def test_with_negative_indices():
    """Test case with negative indices that should be converted to positive."""
    indices = np.array([[0, 1, -1], [1, -1, 3]], dtype=np.int64)
    data_shape = [2, 3, 4]

    model = _make_scatter_nd_model(indices, data_shape)
    graph = OnnxGraph(model)

    # Apply pass
    pm = PassManager(["scatter_nd_indices_to_positive"])
    graph = pm.optimize(graph, strict=True)

    rewritten_indices = evaluate_on_node(
        graph, graph.nodes["scatter_nd/indices_pos"]["pb"]
    )

    # Expected result: [[0, 1, 3], [1, 2, 3]]
    # (since -1 in dim 2 becomes 3, and -1 in dim 1 becomes 2)
    assert rewritten_indices is not None
    expected_indices = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int64)
    assert np.array_equal(rewritten_indices, expected_indices)
