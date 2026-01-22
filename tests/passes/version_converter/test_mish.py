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

from onnxifier.evaluator import Evaluator
from onnxifier.graph import OnnxGraph
from onnxifier.passes.version_converter.downgrade import downgrade_op_version


def _make_mish_op18():
    node = make_node("Mish", ["X"], ["Y"], "mish")
    graph = make_graph(
        [node],
        "graph",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 8, 24, 24])],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, "C", "H", "W"])],
    )
    model = make_model(graph, ir_version=onnx.IR_VERSION)
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def test_downgrade_mish_18_to_17():
    graph = _make_mish_op18()
    runner_ref = Evaluator(graph.model, "onnx")
    x = np.random.uniform(-1, 1, size=[1, 8, 24, 24]).astype(np.float32)
    y_ref = runner_ref(["Y"], {"X": x})[0]

    graph = downgrade_op_version(graph, 17)
    assert len(graph) > 1
    assert graph.opset_version == 17
    onnx.checker.check_model(graph.model, True)
    runner = Evaluator(graph.model, "onnx")
    y = runner(["Y"], {"X": x})[0]

    assert np.allclose(y, y_ref)
