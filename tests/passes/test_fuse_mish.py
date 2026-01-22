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
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _make_graph():
    ide = make_node("Identity", ["x"], ["id"])
    softp = make_node("Softplus", ["id"], ["softp"])
    tanh = make_node("Tanh", ["softp"], ["tanh"])
    mul = make_node("Mul", ["id", "tanh"], ["mul"])
    graph = make_graph(
        [ide, softp, tanh, mul],
        "test",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, (1, 16, 24, 24))],
        [make_tensor_value_info("mul", onnx.TensorProto.FLOAT, (1, 16, 24, 24))],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_fuse_mish():
    graph = _make_graph()
    runner1 = Evaluator(graph.model, "OnnxRuntime")

    pm = PassManager(["fuse_mish"])
    graph = pm.optimize(graph, strict=True)

    assert len(graph) == 2
    for node in graph:
        if graph.nodes[node]["pb"].op_type != "Identity":
            assert graph.nodes[node]["pb"].op_type == "Mish"

    # only onnxruntime can infer functions
    runner2 = Evaluator(graph.model, "OnnxRuntime")

    x = np.random.randn(1, 16, 24, 24).astype(np.float32)
    y1 = runner1([], {"x": x})[0]
    y2 = runner2([], {"x": x})[0]
    assert np.allclose(y1, y2)
