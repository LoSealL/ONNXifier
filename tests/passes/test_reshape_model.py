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

import onnx
from onnx.helper import make_tensor_value_info

from onnxifier import OnnxGraph, PassManager


def _build_graph():
    node = onnx.helper.make_node("Identity", ["x"], ["y"])
    graph = onnx.helper.make_graph(
        [node],
        "test_graph",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["n", "c", "h", "w"])],
        [make_tensor_value_info("y", onnx.TensorProto.FLOAT, ["n", "c", "h", "w"])],
    )
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_reshape_model():
    graph = _build_graph()
    config = {"reshape_model": {"shape_info": {"x": [1, 2, 3, 4], "y": [1, 2, 3, 4]}}}
    pm = PassManager(["reshape_model"], configs=config)
    graph = pm.optimize(graph, strict=True)
    assert graph.tensor_shape("x") == [1, 2, 3, 4]
    assert graph.tensor_shape("y") == [1, 2, 3, 4]


def test_reset_model_batch():
    graph = _build_graph()
    config = {
        "reshape_model": {"shape_info": {"x": [1, 2, 3, 4], "y": [1, 2, 3, 4]}},
        "reset_model_batch": {"batch": 8},
    }
    pm = PassManager(["reshape_model", "reset_model_batch"], configs=config)
    graph = pm.optimize(graph, strict=True)
    assert graph.tensor_shape("x") == [8, 2, 3, 4]
    assert graph.tensor_shape("y") == [8, 2, 3, 4]
