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
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _build_graph():
    conv = make_node("Conv", ["X", "W", "B"], ["Y"], name="conv", auto_pad="SAME_UPPER")
    relu = make_node("Relu", ["Y"], ["Z"], name="relu")
    split = make_node("Split", ["Z"], ["Z1", "Z2"], name="split", axis=1, num_outputs=2)
    graph = make_graph(
        [conv, relu, split],
        "test",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 32, 8, 8])],
        [
            make_tensor_value_info("Z1", onnx.TensorProto.FLOAT, [1, 16, 8, 8]),
            make_tensor_value_info("Z2", onnx.TensorProto.FLOAT, [1, 16, 8, 8]),
        ],
        [
            from_array(np.random.randn(32, 32, 3, 3).astype(np.float32), "W"),
            from_array(np.random.randn(32).astype(np.float32), "B"),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return model


def test_eliminate_slice_after_conv():
    graph = OnnxGraph(_build_graph())
    run1 = Evaluator(graph.model)
    x = np.random.randn(1, 32, 8, 8).astype(np.float32)
    z01, z02 = run1(["Z1", "Z2"], {"X": x})

    pm = PassManager(["eliminate_slice_after_conv", "eliminate_dead_nodes"])
    graph = pm.optimize(graph, strict=True)
    run2 = Evaluator(graph.model)
    z11, z12 = run2(["Z1", "Z2"], {"X": x})

    np.testing.assert_allclose(z01, z11)
    np.testing.assert_allclose(z02, z12)
