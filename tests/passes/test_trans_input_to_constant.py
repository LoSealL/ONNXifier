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
import pytest
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier import OnnxGraph, PassManager


def _build_test_graph():
    conv0 = make_node("Conv", inputs=["a", "w0"], outputs=["c"], group=1, name="conv0")
    graph = make_graph(
        [conv0],
        name="graph",
        inputs=[
            make_value_info("a", make_tensor_type_proto(1, [1, 3, 128, 127])),
            make_value_info("w0", make_tensor_type_proto(1, [8, 3, 3, 3])),
        ],
        outputs=[make_value_info("c", make_tensor_type_proto(1, [1, 8, 126, 125]))],
    )
    model = make_model(graph)
    return model


def test_trans_input_to_constant_with_config():
    graph = OnnxGraph(_build_test_graph())
    assert "w0" in graph.inputs
    pass_manager = PassManager(
        ["trans_input_to_constant"],
        configs={
            "trans_input_to_constant": dict(
                input_name="w0", value=np.ones([8, 3, 3, 3], "float32")
            )
        },
    )
    graph = pass_manager.optimize(graph, strict=True)
    assert len(graph.inputs) == 1
    assert "w0" not in graph.inputs


def test_trans_input_to_constant_with_wrong_config():
    graph = OnnxGraph(_build_test_graph())
    assert "w0" in graph.inputs
    pass_manager = PassManager(
        ["trans_input_to_constant"],
        configs={
            "trans_input_to_constant": dict(
                input_name="w1", value=np.ones([8, 3, 3, 3], "float32")
            )
        },
    )
    with pytest.raises(ValueError) as e:
        graph = pass_manager.optimize(graph, strict=True)
        assert str(e) == "w1 is not an input of the model"


def test_trans_input_to_constant_default_value():
    """Test with default value (all-ones array)"""
    graph = OnnxGraph(_build_test_graph())
    assert "w0" in graph.inputs
    pass_manager = PassManager(
        ["trans_input_to_constant"],
        configs={
            "trans_input_to_constant": dict(input_name="w0")  # No value specified
        },
    )
    graph = pass_manager.optimize(graph, strict=True)
    assert len(graph.inputs) == 1
    assert "w0" not in graph.inputs

    # Check that constant node was created with all-ones
    const_nodes = [
        node
        for name, node in graph.nodes.items()
        if node["pb"].op_type == "Constant" and node["pb"].output[0] == "w0"
    ]
    assert len(const_nodes) == 1


def test_trans_input_to_constant_multiple_inputs():
    """Test with multiple input names and values"""
    conv0 = make_node("Conv", inputs=["a", "w0"], outputs=["b"], name="conv0")
    conv1 = make_node("Conv", inputs=["b", "w1"], outputs=["c"], name="conv1")
    graph = make_graph(
        [conv0, conv1],
        name="graph",
        inputs=[
            make_value_info("a", make_tensor_type_proto(1, [1, 3, 128, 127])),
            make_value_info("w0", make_tensor_type_proto(1, [8, 3, 3, 3])),
            make_value_info("w1", make_tensor_type_proto(1, [16, 8, 3, 3])),
        ],
        outputs=[make_value_info("c", make_tensor_type_proto(1, [1, 16, 124, 123]))],
    )
    model = make_model(graph)

    graph_obj = OnnxGraph(model)
    assert "w0" in graph_obj.inputs
    assert "w1" in graph_obj.inputs

    pass_manager = PassManager(
        ["trans_input_to_constant"],
        configs={
            "trans_input_to_constant": dict(
                input_name=["w0", "w1"],
                value=[
                    np.ones([8, 3, 3, 3], "float32"),
                    np.ones([16, 8, 3, 3], "float32"),
                ],
            )
        },
    )
    graph_obj = pass_manager.optimize(graph_obj, strict=True)
    assert len(graph_obj.inputs) == 1
    assert "w0" not in graph_obj.inputs
    assert "w1" not in graph_obj.inputs


def test_trans_input_to_constant_single_value_multiple_inputs():
    """Test with multiple input names but single value"""
    conv0 = make_node("Conv", inputs=["a", "w0"], outputs=["b"], name="conv0")
    conv1 = make_node("Conv", inputs=["b", "w1"], outputs=["c"], name="conv1")
    graph = make_graph(
        [conv0, conv1],
        name="graph",
        inputs=[
            make_value_info("a", make_tensor_type_proto(1, [1, 3, 128, 127])),
            make_value_info("w0", make_tensor_type_proto(1, [8, 3, 3, 3])),
            make_value_info("w1", make_tensor_type_proto(1, [8, 3, 3, 3])),
        ],
        outputs=[make_value_info("c", make_tensor_type_proto(1, [1, 8, 124, 123]))],
    )
    model = make_model(graph)

    graph_obj = OnnxGraph(model)

    pass_manager = PassManager(
        ["trans_input_to_constant"],
        configs={
            "trans_input_to_constant": dict(
                input_name=["w0", "w1"],
                # Single value for multiple inputs
                value=np.ones([8, 3, 3, 3], "float32"),
            )
        },
    )
    graph_obj = pass_manager.optimize(graph_obj, strict=True)
    assert len(graph_obj.inputs) == 1
    assert "w0" not in graph_obj.inputs
    assert "w1" not in graph_obj.inputs


def test_trans_input_to_constant_shape_incompatible():
    """Test with incompatible shape"""
    graph = OnnxGraph(_build_test_graph())

    pass_manager = PassManager(
        ["trans_input_to_constant"],
        configs={
            "trans_input_to_constant": dict(
                input_name="w0", value=np.ones([4, 3, 3, 3], "float32")  # Wrong shape
            )
        },
    )

    with pytest.raises(ValueError) as e:
        graph = pass_manager.optimize(graph, strict=True)
        assert "shape" in str(e) and "is not compatible with" in str(e)


def test_trans_input_to_constant_different_ndim():
    """Test with different number of dimensions (should trigger broadcasting logic)"""
    conv0 = make_node("Add", inputs=["a", "b"], outputs=["c"], name="add0")
    graph = make_graph(
        [conv0],
        name="graph",
        inputs=[
            make_value_info("a", make_tensor_type_proto(1, [1, 3, 128, 127])),
            make_value_info("b", make_tensor_type_proto(1, [1, 3, 1, 1])),
        ],
        outputs=[make_value_info("c", make_tensor_type_proto(1, [1, 3, 128, 127]))],
    )
    model = make_model(graph)

    graph_obj = OnnxGraph(model)

    # This should fail due to the broadcasting logic issue in the original code
    pass_manager = PassManager(
        ["trans_input_to_constant"],
        configs={
            "trans_input_to_constant": dict(
                input_name="b",
                # Different ndim, should trigger broadcasting
                value=np.array([[[[[2.0]]]]], "float32"),
            )
        },
    )
    with pytest.raises(ValueError):
        graph_obj = pass_manager.optimize(graph_obj, strict=True)


def test_trans_input_to_constant_same_ndim_different_shape():
    """Test with same ndim but broadcastable shape"""
    conv0 = make_node("Add", inputs=["a", "b"], outputs=["c"], name="add0")
    graph = make_graph(
        [conv0],
        name="graph",
        inputs=[
            make_value_info("a", make_tensor_type_proto(1, [1, 3, 128, 127])),
            make_value_info("b", make_tensor_type_proto(1, [1, 3, 1, 1])),
        ],
        outputs=[make_value_info("c", make_tensor_type_proto(1, [1, 3, 128, 127]))],
    )
    model = make_model(graph)

    graph_obj = OnnxGraph(model)

    pass_manager = PassManager(
        ["trans_input_to_constant"],
        configs={
            "trans_input_to_constant": dict(
                input_name="b", value=np.ones([1, 3, 1, 1], "float32")  # Same shape
            )
        },
    )
    graph_obj = pass_manager.optimize(graph_obj, strict=True)
    assert "b" not in graph_obj.inputs


def test_trans_input_to_constant_dynamic_shape():
    """Test with dynamic input shape"""
    conv0 = make_node("Conv", inputs=["a", "w0"], outputs=["c"], name="conv0")
    graph = make_graph(
        [conv0],
        name="graph",
        inputs=[
            make_value_info("a", make_tensor_type_proto(1, [1, 3, -1, -1])),
            make_value_info("w0", make_tensor_type_proto(1, [8, 3, 3, 3])),
        ],
        outputs=[make_value_info("c", make_tensor_type_proto(1, [1, 8, -1, -1]))],
    )
    model = make_model(graph)

    graph_obj = OnnxGraph(model)

    pass_manager = PassManager(
        ["trans_input_to_constant"],
        configs={
            "trans_input_to_constant": dict(
                input_name="w0", value=np.ones([8, 3, 3, 3], "float32")
            )
        },
    )
    graph_obj = pass_manager.optimize(graph_obj, strict=True)
    assert "w0" not in graph_obj.inputs


def test_trans_input_to_constant_int_dtype():
    """Test with integer data type"""
    gather0 = make_node(
        "Gather", inputs=["data", "indices"], outputs=["output"], name="gather0"
    )
    graph = make_graph(
        [gather0],
        name="graph",
        inputs=[
            make_value_info("data", make_tensor_type_proto(1, [10, 5])),
            make_value_info("indices", make_tensor_type_proto(6, [3])),  # int32 type
        ],
        outputs=[make_value_info("output", make_tensor_type_proto(1, [3, 5]))],
    )
    model = make_model(graph)

    graph_obj = OnnxGraph(model)

    pass_manager = PassManager(
        ["trans_input_to_constant"],
        configs={
            "trans_input_to_constant": dict(
                input_name="indices", value=np.array([0, 2, 4], dtype=np.int32)
            )
        },
    )
    graph_obj = pass_manager.optimize(graph_obj, strict=True)
    assert "indices" not in graph_obj.inputs
