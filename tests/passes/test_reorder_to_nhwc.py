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
import pytest
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import ONNXIFIER_OPSET, OnnxGraph, PassManager


def _build_graph_4d():
    conv = make_node("Convolution", ["x", "w", "b"], ["y"])
    graph = make_graph(
        [conv],
        "test",
        [
            make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 224, 224]),
            make_tensor_value_info("w", onnx.TensorProto.FLOAT, [64, 3, 7, 7]),
            make_tensor_value_info("b", onnx.TensorProto.FLOAT, [64]),
        ],
        [
            make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 64, 218, 218]),
        ],
        [
            onnx.numpy_helper.from_array(
                np.random.randn(64, 3, 7, 7).astype(np.float32), "w"
            ),
            onnx.numpy_helper.from_array(np.random.randn(64).astype(np.float32), "b"),
        ],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])
    return model


def _build_graph_5d():
    conv = make_node("Convolution", ["x", "w", "b"], ["y"])
    graph = make_graph(
        [conv],
        "test",
        [
            make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 16, 224, 224]),
            make_tensor_value_info("w", onnx.TensorProto.FLOAT, [64, 3, 3, 1, 1]),
            make_tensor_value_info("b", onnx.TensorProto.FLOAT, [64]),
        ],
        [
            make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 64, 14, 224, 224]),
        ],
        [
            onnx.numpy_helper.from_array(
                np.random.randn(64, 3, 3, 1, 1).astype(np.float32), "w"
            ),
            onnx.numpy_helper.from_array(np.random.randn(64).astype(np.float32), "b"),
        ],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])
    return model


def test_reorder_to_nhwc():
    graph = OnnxGraph(_build_graph_4d())
    pm = PassManager(["reorder_to_nhwc"])
    graph = pm.optimize(graph, strict=True)
    assert graph.tensor_shape("x_nhwc") == [1, 224, 224, 3]


def test_reorder_to_nhwc_failure_case():
    graph = OnnxGraph(_build_graph_5d())
    pm = PassManager(["reorder_to_nhwc"])
    graph = pm.optimize(graph, strict=True)
    with pytest.raises(ValueError):
        graph.tensor_shape("x_nhwc")
    assert graph.tensor_shape("x") == [1, 3, 16, 224, 224]
