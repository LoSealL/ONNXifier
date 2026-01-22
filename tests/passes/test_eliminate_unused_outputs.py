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
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import OnnxGraph, PassManager


def _build_graph():
    node = make_node("Add", ["input1", "input2"], ["output1"])
    graph = make_graph(
        [node],
        "test_graph",
        [
            make_tensor_value_info("input1", onnx.TensorProto.FLOAT, [1, 2, 3]),
            make_tensor_value_info("input2", onnx.TensorProto.FLOAT, [1, 2, 3]),
            make_tensor_value_info("input3", onnx.TensorProto.FLOAT, [1, 2, 3]),
        ],
        [
            make_tensor_value_info("output1", onnx.TensorProto.FLOAT, [1, 2, 3]),
            make_tensor_value_info("output2", onnx.TensorProto.FLOAT, [1, 2, 3]),
            make_tensor_value_info("output3", onnx.TensorProto.FLOAT, [1, 2, 3]),
        ],
    )
    model = make_model(graph)
    return model


def test_eliminate_unused_outputs():
    graph = OnnxGraph(_build_graph())
    assert len(graph.output) == 3
    assert len(graph.outputs) == 3
    pm = PassManager(["eliminate_unused_outputs"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.output) == 1
    assert len(graph.outputs) == 1
