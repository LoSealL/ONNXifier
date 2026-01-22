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
    identity = make_node("Identity", ["x"], ["y"], "identity")
    relu = make_node("Relu", ["y"], ["z"], "relu")
    graph = make_graph(
        [identity, relu],
        "graph",
        [],
        [make_value_info("z", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [
            numpy_helper.from_array(
                np.random.randn(1, 3, 256, 256).astype(np.float32), "x"
            )
        ],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def _build_graph2():
    identity = make_node("Identity", ["x"], ["y"], "identity")
    relu1 = make_node("Relu", ["y"], ["z1"], "relu1")
    relu2 = make_node("Relu", ["y"], ["z2"], "relu2")
    relu3 = make_node("Relu", ["y"], ["z3"], "relu3")
    relu4 = make_node("Relu", ["y"], ["z4"], "relu4")
    graph = make_graph(
        [identity, relu1, relu2, relu3, relu4],
        "graph",
        [],
        [
            make_value_info("z1", make_tensor_type_proto(1, [1, 3, 256, 256])),
            make_value_info("z2", make_tensor_type_proto(1, [1, 3, 256, 256])),
            make_value_info("z3", make_tensor_type_proto(1, [1, 3, 256, 256])),
            make_value_info("z4", make_tensor_type_proto(1, [1, 3, 256, 256])),
        ],
        [
            numpy_helper.from_array(
                np.random.randn(1, 3, 256, 256).astype(np.float32), "x"
            )
        ],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def test_replicate_identity_constant_1fanout():
    model = _build_graph1()
    graph = OnnxGraph(model)
    pm = PassManager(["replicate_identity_constant"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 2
    assert "identity" not in graph


def test_replicate_identity_constant_4fanout():
    model = _build_graph2()
    graph = OnnxGraph(model)
    pm = PassManager(["replicate_identity_constant"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 8
    assert "identity" not in graph
