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
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _build_graph():
    resize = make_node(
        "Resize", ["x", "", "scales"], ["y"], axes=[2, 3, 4], name="resize"
    )
    graph = make_graph(
        [resize],
        "test",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 64, 128, 256])],
        [make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3, 32, 32, 32])],
        [from_array(np.array([0.5, 0.25, 0.125], np.float32), "scales")],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    return OnnxGraph(model)


def test_resize_remove_axes():
    graph = _build_graph()
    try:
        run1 = Evaluator(graph.model, "OnnxRuntime")
        x = np.random.randn(1, 3, 64, 128, 256).astype(np.float32)
        y1 = run1(["y"], {"x": x})[0]
        assert y1.shape == (1, 3, 32, 32, 32)

        pm = PassManager(["resize_remove_axes"])
        graph = pm.optimize(graph, strict=True)
        run2 = Evaluator(graph.model, "OnnxRuntime")
        y2 = run2(["y"], {"x": x})[0]
        assert np.allclose(y1, y2)
    except ImportError:
        pytest.skip("onnxruntime not installed")
