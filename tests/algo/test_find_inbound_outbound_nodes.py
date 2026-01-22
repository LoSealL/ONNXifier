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
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph
from onnxifier.algo.subgraph import find_inbound_nodes, find_outbound_nodes


def _make_graph():
    sin1 = make_node("Sin", ["X1"], ["S1"], name="sin1")
    sin2 = make_node("Sin", ["X2"], ["S2"], name="sin2")
    act2 = make_node("Relu", ["S2"], ["A2"], name="act2")
    cat1 = make_node("Concat", ["S1", "A2"], ["C1"], name="cat1")
    sin3 = make_node("Sin", ["C1"], ["S3"], name="sin3")
    act3 = make_node("Relu", ["S3"], ["A3"], name="act3")

    graph = make_graph(
        [sin1, sin2, act2, cat1, sin3, act3],
        "test_graph",
        [
            make_tensor_value_info("X1", onnx.TensorProto.FLOAT, [1, 2, 3, 4]),
            make_tensor_value_info("X2", onnx.TensorProto.FLOAT, [1, 2, 3, 4]),
        ],
        [
            make_tensor_value_info("A3", onnx.TensorProto.FLOAT, [1, 2, 3, 4]),
        ],
    )
    return make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )


def test_find_inbound_nodes():
    graph = OnnxGraph(_make_graph())

    nodes, startpoints = find_inbound_nodes(graph, graph.nodes["act3"]["pb"], {"Sin"})
    assert len(nodes) == 0
    assert startpoints[0].name == "sin3"

    nodes, startpoints = find_inbound_nodes(graph, graph.nodes["sin3"]["pb"], {"Sin"})
    assert len(nodes) == 2
    assert nodes[0].name == "cat1"
    assert nodes[1].name == "act2"
    assert startpoints[0].name == "sin1"
    assert startpoints[1].name == "sin2"


def test_find_outbound_nodes():
    graph = OnnxGraph(_make_graph())

    nodes, endpoints = find_outbound_nodes(graph, graph.nodes["sin1"]["pb"], {"Sin"})
    assert len(nodes) == 1
    assert nodes[0].name == "cat1"
    assert endpoints[0].name == "sin3"
