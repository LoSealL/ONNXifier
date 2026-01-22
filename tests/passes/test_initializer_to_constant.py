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
from onnx.helper import make_graph, make_node, make_tensor_value_info

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _build_graph():
    add1 = make_node("Add", ["x", "add1"], ["z"], name="add1")
    graph = make_graph(
        [add1],
        "test",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1000])],
        [make_tensor_value_info("z", onnx.TensorProto.FLOAT, [1, 1000])],
        initializer=[
            # this is a legal name in ONNX
            onnx.numpy_helper.from_array(
                np.random.randn(1, 1000).astype(np.float32), "add1"
            )
        ],
    )
    model = onnx.helper.make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, full_check=False)
    return model


def test_initializer_to_constant_duplicate_names():
    model = _build_graph()
    graph = OnnxGraph(model)
    pm = PassManager(["initializer_to_constant"])
    graph = pm.optimize(graph, strict=True)

    assert len(graph.initializer) == 0
    assert len(graph) == 2

    # Check the graph can be evaluated
    Evaluator(graph.model)([], {"x": np.random.randn(1, 1000).astype(np.float32)})
