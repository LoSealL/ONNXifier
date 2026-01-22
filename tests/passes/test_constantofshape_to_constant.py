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
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import OnnxGraph, PassManager


def _build_graph():
    cos = make_node("ConstantOfShape", ["shape"], ["output"])
    graph = make_graph(
        [cos],
        "test_graph",
        [],
        [make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 3])],
        initializer=[
            onnx.numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "shape")
        ],
    )
    model = make_model(graph)
    onnx.checker.check_model(model, True)
    return model


def test_constantofshape_to_constant_from_initializer():
    model = _build_graph()
    graph = OnnxGraph(model)
    pm = PassManager(["constantofshape_to_constant"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 1
    for node in graph.nodes:
        node = graph.nodes[node]["pb"]
        assert node.op_type == "Constant"


def test_constantofshape_to_constant_from_constant():
    model = _build_graph()
    graph = OnnxGraph(model)
    pm = PassManager(["initializer_to_constant", "constantofshape_to_constant"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 2
    for node in graph.nodes:
        node = graph.nodes[node]["pb"]
        assert node.op_type == "Constant"
