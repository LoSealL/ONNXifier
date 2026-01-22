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

# pylint: disable=missing-function-docstring

import numpy as np
import onnx
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, PassManager
from onnxifier.evaluator import Evaluator
from onnxifier.graph import OnnxGraph


def _make_col2im_op18(
    image_shape, block_shape, dilations=None, pads=None, strides=None
):
    node = make_node(
        "Col2Im",
        ["input", "image_shape", "block_shape"],
        ["output"],
        "col2im",
        dilations=dilations,
        pads=pads,
        strides=strides,
    )
    graph = make_graph(
        [node],
        "graph",
        [
            make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 32, "L"]),
        ],
        [make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, "C", "H", "W"])],
        [
            from_array(np.array(image_shape, np.int64), "image_shape"),
            from_array(np.array(block_shape, np.int64), "block_shape"),
        ],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def test_col2im_to_depthtospace():
    graph = _make_col2im_op18(image_shape=[24, 24], block_shape=[2, 2], strides=[2, 2])
    runner1 = Evaluator(graph.model)
    x = np.random.randn(1, 32, 144).astype("float32")
    [y_ref] = runner1(["output"], input=x)

    pm = PassManager(["col2im_to_depthtospace"])
    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model)
    [y] = runner2(["output"], input=x)

    assert y_ref.shape == (1, 8, 24, 24)
    assert y.shape == (1, 8, 24, 24)
    assert np.allclose(y, y_ref)
