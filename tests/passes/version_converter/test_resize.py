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
import pytest
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET
from onnxifier.evaluator import Evaluator
from onnxifier.graph import OnnxGraph
from onnxifier.passes.version_converter.downgrade import downgrade_op_version


def _make_resize_op19(**attrs):
    node = make_node(
        "Resize",
        ["X", "", "", "sizes"],
        ["Y"],
        "resize",
        axes=[2, 3],
        **attrs,
    )
    graph = make_graph(
        [node],
        "graph",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 24, 16])],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3, "H", "W"])],
        [from_array(np.array([12, 12], np.int64), "sizes")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def test_downgrade_op_version_api_exceptions():
    for i in range(22):
        if i in (13, 15, 17, 19):
            continue
        with pytest.raises(NotImplementedError):
            downgrade_op_version(None, i)


def test_resize_19_to_13_antialias():
    graph = _make_resize_op19(antialias=1, mode="cubic")
    with pytest.raises(ValueError):
        downgrade_op_version(graph, 13)
    graph = _make_resize_op19(antialias=1, mode="linear")
    with pytest.raises(ValueError):
        downgrade_op_version(graph, 13)


def test_resize_19_to_13():
    graph = _make_resize_op19()
    runner_ref = Evaluator(graph.model, "onnx")
    x = np.random.uniform(0, 255, size=[1, 3, 24, 16]).astype(np.float32)
    y_ref = runner_ref(["Y"], {"X": x})[0]

    graph = downgrade_op_version(graph, 13)
    onnx.checker.check_model(graph.model, True)
    runner = Evaluator(graph.model, "onnx")
    y_dut = runner(["Y"], {"X": x})[0]

    assert y_ref.shape == y_dut.shape
    assert y_ref.dtype == y_dut.dtype
    assert np.allclose(y_ref, y_dut)


def test_resize_19_to_17_keep_ratio_not_larger():
    graph = _make_resize_op19(keep_aspect_ratio_policy="not_larger")
    runner_ref = Evaluator(graph.model, "onnx")
    x = np.random.uniform(0, 255, size=[1, 3, 24, 16]).astype(np.float32)
    y_ref = runner_ref(["Y"], {"X": x})[0]

    graph = downgrade_op_version(graph, 17)
    onnx.checker.check_model(graph.model, True)
    runner = Evaluator(graph.model, "onnx")
    y_dut = runner(["Y"], {"X": x})[0]

    assert y_ref.shape == y_dut.shape
    assert y_ref.dtype == y_dut.dtype
    assert np.allclose(y_ref, y_dut)


def test_resize_19_to_17_keep_ratio_not_smaller():
    graph = _make_resize_op19(keep_aspect_ratio_policy="not_smaller")
    runner_ref = Evaluator(graph.model, "onnx")
    x = np.random.uniform(0, 255, size=[1, 3, 24, 16]).astype(np.float32)
    y_ref = runner_ref(["Y"], {"X": x})[0]

    graph = downgrade_op_version(graph, 17)
    onnx.checker.check_model(graph.model, True)
    runner = Evaluator(graph.model, "onnx")
    y_dut = runner(["Y"], {"X": x})[0]

    assert y_ref.shape == y_dut.shape
    assert y_ref.dtype == y_dut.dtype
    assert np.allclose(y_ref, y_dut)
