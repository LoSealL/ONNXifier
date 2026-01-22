"""
Copyright (C) 2024 The ONNXIFIER Authors.

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
    get_attribute_value,
    make_graph,
    make_model,
    make_node,
    make_tensor_value_info,
)

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _make_graph():
    softmax = make_node("Softmax", ["X"], ["Y"], axis=1)
    graph = make_graph(
        [softmax],
        "test",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 16, 8, 40])],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 16, 8, 40])],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_transpose_softmax_to_last_dim():
    graph = _make_graph()
    runner1 = Evaluator(graph.model)

    pm = PassManager(["transpose_softmax_to_last_dim"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph) == 3
    for node in graph:
        node_pb: onnx.NodeProto = graph.nodes[node]["pb"]
        if node_pb.op_type == "Softmax":
            assert get_attribute_value(node_pb.attribute[0]) == -1

    runner2 = Evaluator(graph.model)

    x = np.random.randn(1, 16, 8, 40).astype(np.float32)
    y1 = runner1([], {"X": x})[0]
    y2 = runner2([], {"X": x})[0]
    assert np.allclose(y1, y2)
