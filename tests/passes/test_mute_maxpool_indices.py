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

import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager


def _build_graph(use_indices):
    maxpool = make_node(
        "MaxPool",
        ["x"],
        ["y", "indices"],
        kernel_shape=[2, 2],
        auto_pad="SAME_UPPER",
        strides=[2, 2],
        name="maxpool",
    )
    relu1 = make_node("Relu", ["y"], ["y1"])
    relu2 = make_node("Relu", ["y"], ["y2"])
    add = make_node("Add", ["y1", "y2"], ["z"])
    identity = make_node("Identity", ["indices"], ["id"])
    graph = make_graph(
        [maxpool, relu1, relu2, add] + ([identity] if use_indices else []),
        "test",
        [make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 5])],
        [make_tensor_value_info("z", TensorProto.FLOAT, [1, 3, "H", "W"])],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_mute_maxpool_indices():
    graph = _build_graph(use_indices=False)
    pm = PassManager(["mute_maxpool_indices"])
    graph = pm.optimize(graph, strict=True)

    assert len(graph.nodes["maxpool"]["pb"].output) == 1


def test_mute_maxpool_indices_false_case():
    graph = _build_graph(use_indices=True)
    pm = PassManager(["mute_maxpool_indices"])
    graph = pm.optimize(graph, strict=True)

    assert len(graph.nodes["maxpool"]["pb"].output) == 2
