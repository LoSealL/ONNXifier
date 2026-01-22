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
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET
from onnxifier.evaluator import Evaluator
from onnxifier.graph import OnnxGraph
from onnxifier.passes.version_converter.downgrade import downgrade_op_version


def _make_bitwise_not_op19(elem_type):
    node = make_node(
        "BitwiseNot",
        ["X"],
        ["Y"],
        "bitwise_not",
    )
    graph = make_graph(
        [node],
        "graph",
        [make_tensor_value_info("X", elem_type, [1, 8, 24, 24])],
        [make_tensor_value_info("Y", elem_type, [1, "C", "H", "W"])],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def test_downgrade_avgpool_19_to_17_i32():
    graph = _make_bitwise_not_op19(onnx.TensorProto.INT32)
    runner_ref = Evaluator(graph.model, "onnx")
    x = np.random.uniform(0, 255, size=[1, 8, 24, 24]).astype(np.int32)
    y_ref = runner_ref(["Y"], {"X": x})[0]

    graph = downgrade_op_version(graph, 17)
    assert graph.opset_version == 17
    runner = Evaluator(graph.model, "onnx")
    y = runner(["Y"], {"X": x})[0]

    assert np.allclose(y, y_ref)


def test_downgrade_avgpool_19_to_17_u8():
    graph = _make_bitwise_not_op19(onnx.TensorProto.UINT8)
    runner_ref = Evaluator(graph.model, "onnx")
    x = np.random.uniform(0, 255, size=[1, 8, 24, 24]).astype(np.uint8)
    y_ref = runner_ref(["Y"], {"X": x})[0]

    graph = downgrade_op_version(graph, 17)
    assert graph.opset_version == 17
    runner = Evaluator(graph.model, "onnx")
    y = runner(["Y"], {"X": x})[0]

    assert np.allclose(y, y_ref)
