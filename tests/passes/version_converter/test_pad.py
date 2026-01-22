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

# pylint: disable=missing-function-docstring

import numpy as np
import onnx
import pytest
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array

from onnxifier.evaluator import Evaluator
from onnxifier.graph import OnnxGraph
from onnxifier.passes.version_converter.downgrade import downgrade_op_version


def _make_pad_op19(pads, axes, mode="constant"):
    node = make_node("Pad", ["X", "pads", "", "axes"], ["Y"], "pad", mode=mode)
    graph = make_graph(
        [node],
        "graph",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 8, 24, 24])],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, "C", "H", "W"])],
        [
            from_array(np.array(pads, np.int64), "pads"),
            from_array(np.array(axes, np.int64), "axes"),
        ],
    )
    model = make_model(graph, ir_version=onnx.IR_VERSION)
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def test_downgrade_pad_19_to_13():
    graph = _make_pad_op19([1, 1, 1, 1], axes=[2, 3])
    runner_ref = Evaluator(graph.model, "onnx")
    x = np.random.uniform(-1, 1, size=[1, 8, 24, 24]).astype(np.float32)
    y_ref = runner_ref(["Y"], {"X": x})[0]

    graph = downgrade_op_version(graph, 13)
    assert graph.opset_version == 13
    onnx.checker.check_model(graph.model, True)
    runner = Evaluator(graph.model, "onnx")
    y = runner(["Y"], {"X": x})[0]

    assert np.allclose(y, y_ref)


def test_downgrade_pad_19_mode_exception():
    graph = _make_pad_op19([1, 1, 1, 1], axes=[2, 3], mode="wrap")
    with pytest.raises(ValueError):
        downgrade_op_version(graph, 13)
