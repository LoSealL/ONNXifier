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

from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_type_proto,
    make_tensor_value_info,
    make_value_info,
)

from onnxifier import ONNXIFIER_OPSET, PassManager
from onnxifier.graph import OnnxGraph


def _build_graph_double_identities():
    identity1 = make_node("Identity", ["x"], ["y"], "identity1")
    identity2 = make_node("Identity", ["y"], ["z"], "identity2")
    relu = make_node("Relu", ["z"], ["w"], "relu")
    graph = make_graph(
        [identity1, identity2, relu],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [make_value_info("w", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def _build_graph_output_identity():
    relu = make_node("Relu", ["x"], ["y"], "relu")
    identity = make_node("Identity", ["y"], ["z"], "identity")
    graph = make_graph(
        [relu, identity],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [make_value_info("z", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def test_eliminate_double_identities():
    model = _build_graph_double_identities()
    graph = OnnxGraph(model)
    pm = PassManager(["eliminate_identity"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 1
    assert "identity1" not in graph
    assert "identity2" not in graph
    assert "x" in graph.inputs
    assert "w" in graph.outputs
    check_model(graph.model, True)


def test_eliminate_output_identity():
    model = _build_graph_output_identity()
    graph = OnnxGraph(model)
    pm = PassManager(["eliminate_identity"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 1
    assert "identity" not in graph
    assert "x" in graph.inputs
    assert "z" in graph.outputs
    check_model(graph.model, True)


def test_eliminate_identity_with_siblings():
    nodes = [
        make_node("Relu", ["x"], ["y"], "relu"),
        make_node("Identity", ["y"], ["z"], "identity"),
        make_node("Sin", ["y"], ["w"], "sin"),
    ]
    model = make_model(
        make_graph(
            nodes,
            "graph",
            [make_tensor_value_info("x", TensorProto.FLOAT, [32])],
            [
                make_tensor_value_info("z", TensorProto.FLOAT, [32]),
                make_tensor_value_info("w", TensorProto.FLOAT, [32]),
            ],
        ),
        opset_imports=[ONNXIFIER_OPSET],
    )
    graph = OnnxGraph(model)
    pm = PassManager(["eliminate_identity"])
    graph = pm.optimize(graph, strict=True)
    assert len(graph.nodes) == 2
    assert "identity" not in graph
    assert "x" in graph.inputs
    assert "z" in graph.outputs
    assert "w" in graph.outputs
    check_model(graph.model, True)
