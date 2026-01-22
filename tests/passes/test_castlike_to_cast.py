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
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager


def _make_graph():
    castlike = make_node("CastLike", ["input", "other"], ["output"], name="castlike")
    graph = make_graph(
        [castlike],
        "test",
        [make_tensor_value_info("input", onnx.TensorProto.UINT8, [1, 2, 3, 4])],
        [make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2, 3, 4])],
        [from_array(np.ones([1], dtype=np.float32), "other")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def test_castlike_to_cast():
    graph = _make_graph()
    pm = PassManager(["castlike_to_cast"])
    graph = pm.optimize(graph, strict=True)

    assert graph.nodes["castlike"]["pb"].op_type == "Cast"
    onnx.checker.check_model(graph.model, True)
