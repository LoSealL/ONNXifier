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

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET
from onnxifier.evaluator import Evaluator
from onnxifier.graph import OnnxGraph
from onnxifier.passes.version_converter.downgrade import downgrade_op_version


def _make_reduce_op_op19(op_type, **attr):
    node = make_node(op_type, ["X", "axes"], ["Y"], "reduce", **attr)
    graph = make_graph(
        [node],
        "graph",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 8, 24, 24])],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, "C", "H", "W"])],
        [from_array(np.array([2, 3], np.int64), "axes")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def test_downgrade_reduce_mean_19_to_17():
    graph = _make_reduce_op_op19("ReduceMean")
    runner_ref = Evaluator(graph.model, "onnx")
    x = np.random.uniform(0, 255, size=[1, 8, 24, 24]).astype(np.float32)
    y_ref = runner_ref(["Y"], {"X": x})[0]

    graph = downgrade_op_version(graph, 17)
    assert graph.opset_version == 17
    onnx.checker.check_model(graph.model, True)
    runner = Evaluator(graph.model, "onnx")
    y = runner(["Y"], {"X": x})[0]

    assert y.dtype == y_ref.dtype
    assert np.allclose(y, y_ref)


def test_downgrade_reduce_mean_19_noop_exception():
    graph = _make_reduce_op_op19("ReduceMean", noop_with_empty_axes=1)
    with pytest.raises(ValueError):
        downgrade_op_version(graph, 17)
