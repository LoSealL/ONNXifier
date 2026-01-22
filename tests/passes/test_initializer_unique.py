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
from onnx.helper import make_graph, make_node, make_tensor_value_info

from onnxifier import OnnxGraph, PassManager


def _build_graph():
    add1 = make_node("Add", ["x", "y"], ["z"])
    add2 = make_node("Add", ["z", "y"], ["o"])
    graph = make_graph(
        [add1, add2],
        "test",
        [
            make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1000]),
        ],
        [make_tensor_value_info("o", onnx.TensorProto.FLOAT, [1, 1000])],
        initializer=[
            onnx.numpy_helper.from_array(
                np.random.randn(1, 1000).astype(np.float32), "y"
            ),
            # duplicated initializer
            onnx.numpy_helper.from_array(
                np.random.randn(1, 1000).astype(np.float32), "y"
            ),
        ],
    )
    model = onnx.helper.make_model(graph)
    with pytest.raises(
        (
            onnx.checker.ValidationError,
            onnx.shape_inference.InferenceError,
        )
    ):
        # check failed due to duplicated initializers
        onnx.checker.check_model(model, full_check=True)
    return model


def test_eliminate_duplicated_initializer():
    graph = OnnxGraph(_build_graph())
    pm = PassManager(["initializer_unique"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.initializer) == 2
    onnx.checker.check_model(graph.model, full_check=True)
