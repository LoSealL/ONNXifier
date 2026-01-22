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
from onnx.helper import make_graph, make_node, make_tensor_value_info
from onnx.reference import ReferenceEvaluator

from onnxifier import OnnxGraph, PassManager


def _build_graph():
    resize = make_node("Resize", ["x", "", "", "sizes"], ["y"])
    graph = make_graph(
        [resize],
        "test",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, [4, 3, 4, 4])],
        [make_tensor_value_info("y", onnx.TensorProto.FLOAT, [4, 3, 8, 8])],
        initializer=[
            onnx.numpy_helper.from_array(
                np.array([4, 3, 8, 8], dtype="int64"), "sizes"
            ),
        ],
    )
    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model, full_check=True)
    return model


def test_change_resize_batch():
    graph = OnnxGraph(_build_graph())
    runner1 = ReferenceEvaluator(graph.model)
    x = np.random.uniform(0, 1, size=[4, 3, 4, 4]).astype(np.float32)
    y1 = runner1.run(None, {"x": x})[0]

    pm = PassManager(
        ["initlializer_to_constant", "resize_move_size_to_scale", "change_resize_batch"]
    )
    graph = pm.optimize(graph, strict=True)
    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x": x})
    assert np.allclose(y1, y2)
