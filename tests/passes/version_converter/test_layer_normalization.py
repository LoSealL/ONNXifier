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
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION
from onnxifier.evaluator import Evaluator
from onnxifier.graph import OnnxGraph
from onnxifier.passes.version_converter.downgrade import downgrade_op_version


def _make_layer_normalization_op17():
    node = make_node(
        "LayerNormalization",
        ["X", "Scale", "B"],
        ["Y", "Mean", "InvStd"],
        "ln",
        axis=1,
    )
    graph = make_graph(
        [node],
        "graph",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 8, 24, 24])],
        [
            make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, "C", "H", "W"]),
            make_tensor_value_info("Mean", onnx.TensorProto.FLOAT, [1, 1, "H", "W"]),
        ],
        initializer=[
            from_array(np.random.rand(1, 8, 1, 1).astype(np.float32), "Scale"),
            from_array(np.random.rand(1, 8, 24, 24).astype(np.float32), "B"),
        ],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[make_operatorsetid("", 17)],
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def test_downgrade_ln_17_to_15():
    graph = _make_layer_normalization_op17()
    runner_ref = Evaluator(graph.model, "onnx")
    x = np.random.uniform(0, 255, size=[1, 8, 24, 24]).astype(np.float32)
    y_ref, mean_ref, inv_std_ref = runner_ref(["Y", "Mean", "InvStd"], {"X": x})

    graph = downgrade_op_version(graph, 15)
    assert graph.opset_version == 15
    runner = Evaluator(graph.model, "onnx")
    y, mean, inv_std = runner(["Y", "Mean", "InvStd"], {"X": x})

    assert np.allclose(y, y_ref)
    assert np.allclose(mean, mean_ref)
    assert np.allclose(inv_std, inv_std_ref)


def test_downgrade_ln_19_to_15():
    graph = _make_layer_normalization_op17()
    graph._model.opset_import[0].version = 19
    runner_ref = Evaluator(graph.model, "onnx")
    x = np.random.uniform(0, 255, size=[1, 8, 24, 24]).astype(np.float32)
    y_ref, mean_ref, inv_std_ref = runner_ref(["Y", "Mean", "InvStd"], {"X": x})

    graph = downgrade_op_version(graph, 15)
    assert graph.opset_version == 15
    runner = Evaluator(graph.model, "onnx")
    y, mean, inv_std = runner(["Y", "Mean", "InvStd"], {"X": x})

    assert np.allclose(y, y_ref)
    assert np.allclose(mean, mean_ref)
    assert np.allclose(inv_std, inv_std_ref)
