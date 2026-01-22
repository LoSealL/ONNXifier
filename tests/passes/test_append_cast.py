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
from onnx import numpy_helper
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier import ONNXIFIER_OPSET, PassManager
from onnxifier.graph import OnnxGraph


def _build_graph1():
    relu = make_node("Relu", ["x"], ["y"], "relu")
    graph = make_graph(
        [relu],
        "graph",
        [],
        [make_value_info("y", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [
            numpy_helper.from_array(
                np.random.randn(1, 3, 256, 256).astype(np.float32), "x"
            )
        ],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def _build_graph2():
    relu = make_node("Relu", ["x"], ["y"], "relu")
    graph = make_graph(
        [relu],
        "graph",
        [],
        [make_value_info("y", make_tensor_type_proto(2, [1, 3, 256, 256]))],
        [
            numpy_helper.from_array(
                np.random.randint(0, 256, size=(1, 3, 256, 256), dtype="uint8"), "x"
            )
        ],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def test_append_cast_float():
    model = _build_graph1()
    graph = OnnxGraph(model)
    pm = PassManager(["append_cast"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 2
    assert "y/cast" in graph


def test_append_cast_uint8():
    model = _build_graph2()
    graph = OnnxGraph(model)
    pm = PassManager(["append_cast"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 2
    assert "y/cast" in graph
