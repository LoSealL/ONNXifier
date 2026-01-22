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
from onnx.helper import (
    make_attribute,
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_type_proto,
    make_tensor_value_info,
    make_value_info,
)

from onnxifier import OnnxGraph
from onnxifier.passes.pattern import (
    GraphPattern,
    InputNodePattern,
    OutputNodePattern,
    SingleNodePattern,
    StartEndPointPattern,
)


def _build_test_graph1():
    conv0 = make_node(
        "Conv",
        inputs=["a", "w0"],
        outputs=["c"],
        group=2,
        name="conv0",
        domain="test.domain",
    )
    conv1 = make_node("Conv", inputs=["c", "w1"], outputs=["d"], group=1, name="conv1")
    graph = make_graph(
        [conv0, conv1],
        name="graph",
        inputs=[make_value_info("a", make_tensor_type_proto(1, [1, 3, 128, 127]))],
        outputs=[make_value_info("d", make_tensor_type_proto(1, None))],
        initializer=[
            make_tensor(
                "w0",
                1,
                [8, 3, 3, 3],
                np.ones([8, 3, 3, 3], "float32").tobytes(),
                raw=True,
            ),
            make_tensor(
                "w1",
                1,
                [8, 8, 3, 3],
                np.ones([8, 8, 3, 3], "float32").tobytes(),
                raw=True,
            ),
        ],
    )
    model = make_model(graph)
    return model


def _build_test_graph2():
    conv0 = make_node("Conv", inputs=["a", "w0"], outputs=["c"], group=2, name="conv0")
    conv1 = make_node("Conv", inputs=["c", "w1"], outputs=["d"], group=1, name="conv1")
    add = make_node("Add", inputs=["c", "d"], outputs=["y"], name="add")
    graph = make_graph(
        [conv0, conv1, add],
        name="graph",
        inputs=[make_value_info("a", make_tensor_type_proto(1, [1, 3, 128, 128]))],
        outputs=[make_value_info("y", make_tensor_type_proto(1, None))],
        initializer=[
            make_tensor(
                "w0",
                1,
                [8, 3, 3, 3],
                np.ones([8, 3, 3, 3], "float32").tobytes(),
                raw=True,
            ),
            make_tensor(
                "w1",
                1,
                [8, 8, 3, 3],
                np.ones([8, 8, 3, 3], "float32").tobytes(),
                raw=True,
            ),
        ],
    )
    model = make_model(graph)
    return model


def test_single_node_match():
    graph = OnnxGraph(_build_test_graph1())
    pattern = SingleNodePattern("Conv")
    nodes = list(pattern.match(graph))
    assert len(nodes) == 2
    for i in nodes:
        assert isinstance(i, onnx.NodeProto)
        assert i.op_type == "Conv"

    pattern = SingleNodePattern("Conv").with_attr("group", 1)
    nodes = list(pattern.match(graph))
    assert len(nodes) == 1
    for i in nodes:
        assert i.attribute[0].i == 1

    pattern1 = SingleNodePattern("Conv").with_attr(make_attribute("group", 1))
    pattern2 = SingleNodePattern("Conv").with_attr("group", 2)
    nodes = list((pattern1 | pattern2).match(graph))
    assert len(nodes) == 2
    for i in nodes:
        assert isinstance(i, onnx.NodeProto)
        assert i.op_type == "Conv"
        assert i.attribute[0].i in (1, 2)

    pattern = SingleNodePattern("Conv").with_attr("dilations")
    nodes = list(pattern.match(graph))
    assert len(nodes) == 0

    pattern = SingleNodePattern().with_name("conv1")
    nodes = list(pattern.match(graph))
    assert len(nodes) == 1
    assert nodes[0].op_type == "Conv"

    pattern = SingleNodePattern("Conv").with_domain("test.domain")
    nodes = list(pattern.match(graph))
    assert len(nodes) == 1

    pattern = SingleNodePattern("Conv").with_domain("*")
    nodes = list(pattern.match(graph))
    assert len(nodes) == 1

    pattern = SingleNodePattern("Conv")
    nodes = list(pattern.match(graph))
    assert len(nodes) == 2


def test_single_node_match_inputs_outputs():
    graph = OnnxGraph(_build_test_graph2())
    pattern = SingleNodePattern("Conv").with_inputs(2)
    nodes = list(pattern.match(graph))
    assert len(nodes) == 1
    assert nodes[0].op_type == "Conv"
    assert nodes[0].name == "conv1"

    pattern = SingleNodePattern("Conv").with_inputs(2, "conv0", "w1")
    nodes = list(pattern.match(graph))
    assert len(nodes) == 1
    assert nodes[0].op_type == "Conv"
    assert nodes[0].name == "conv1"

    pattern = SingleNodePattern("Conv").with_outputs(2, "add", None)
    nodes = list(pattern.match(graph))
    assert len(nodes) == 1
    assert nodes[0].op_type == "Conv"
    assert nodes[0].name == "conv0"

    pattern = SingleNodePattern().with_outputs(1)
    nodes = list(pattern.match(graph))
    assert len(nodes) == 1
    assert nodes[0].op_type == "Conv"
    assert nodes[0].name == "conv1"


def test_single_node_match_order():
    g = make_graph(
        [
            make_node("Relu", ["1"], ["2"], name="relu1"),
            make_node("Relu", ["2"], ["3"], name="relu2"),
            make_node("Relu", ["3"], ["4"], name="relu3"),
        ],
        name="graph",
        inputs=[make_tensor_value_info("1", onnx.TensorProto.FLOAT, [10])],
        outputs=[make_tensor_value_info("4", onnx.TensorProto.FLOAT, [10])],
    )
    model = make_model(g)
    graph = OnnxGraph(model)

    pattern = SingleNodePattern("Relu").with_order("post")
    nodes = list(pattern.match(graph))
    assert nodes[0].name == g.node[2].name
    assert nodes[1].name == g.node[1].name
    assert nodes[2].name == g.node[0].name

    pattern = SingleNodePattern("Relu").with_order("pre")
    nodes = list(pattern.match(graph))
    assert nodes[0].name == g.node[0].name
    assert nodes[1].name == g.node[1].name
    assert nodes[2].name == g.node[2].name


def test_or_pattern():
    graph = OnnxGraph(_build_test_graph1())
    pattern1 = SingleNodePattern("Conv").with_attr("group", 1)
    pattern2 = SingleNodePattern("Conv").with_attr("group", 2)
    nodes = list((pattern1 | pattern2).match(graph))
    assert len(nodes) == 2
    nodes = list((pattern1 + pattern2).match(graph))
    assert len(nodes) == 2
    for i in nodes:
        assert isinstance(i, onnx.NodeProto)
        assert i.op_type == "Conv"
        assert i.attribute[0].i in (1, 2)


def test_subgraph_match():
    graph = OnnxGraph(_build_test_graph1())
    pattern = GraphPattern().add_edge(
        SingleNodePattern("Conv"), SingleNodePattern("Conv")
    )
    nodes = list(pattern.match(graph))
    assert len(nodes) == 1
    assert len(nodes[0]) == 2

    pattern = GraphPattern(pattern)
    nodes = list(pattern.match(graph))
    assert len(nodes) == 1
    assert len(nodes[0]) == 2


def test_input_node_match():
    graph = OnnxGraph(_build_test_graph1())
    pattern = InputNodePattern()
    nodes = list(pattern.match(graph))
    assert len(nodes) == 1
    assert nodes[0].name == "conv0"


def test_output_node_match():
    graph = OnnxGraph(_build_test_graph1())
    pattern = OutputNodePattern()
    nodes = list(pattern.match(graph))
    assert len(nodes) == 1
    assert nodes[0].name == "conv1"


def test_start_end_point_match():
    graph = OnnxGraph(_build_test_graph2())
    pattern = StartEndPointPattern(
        SingleNodePattern(op_name="conv0"), SingleNodePattern(op_name="add")
    )
    nodes_list = list(pattern.match(graph))
    assert len(nodes_list) == 1
    for nodes in nodes_list:
        assert any(i.name == "conv1" for i in nodes)

    pattern_a = GraphPattern()
    pattern_a.add_edge(SingleNodePattern("Conv"), SingleNodePattern("Conv"))
    pattern = StartEndPointPattern(pattern_a, SingleNodePattern("Add"))
    nodes_list = list(pattern.match(graph))
    assert len(nodes_list) == 1
    assert len(nodes_list[0]) == 3

    pattern_b = GraphPattern()
    pattern_b.add_edge(
        SingleNodePattern().with_name("conv1"), SingleNodePattern().with_name("add")
    )
    pattern = StartEndPointPattern(SingleNodePattern("Conv"), pattern_b)
    nodes_list = list(pattern.match(graph))
    assert len(nodes_list) == 1
    assert len(nodes_list[0]) == 3


def test_single_match_pattern_with_specify_node_names():
    graph = OnnxGraph(_build_test_graph2())
    pattern = SingleNodePattern()
    nodes = list(pattern.match(graph, specify_node_names={"add", "conv1"}))
    assert len(nodes) == 2
    assert {n.name for n in nodes} == {"add", "conv1"}


def test_graph_pattern_with_specify_node_names():
    graph = OnnxGraph(_build_test_graph2())
    pattern = GraphPattern().add_edge(
        SingleNodePattern("Conv"), SingleNodePattern("Conv")
    )
    nodes = list(pattern.match(graph, specify_node_names={"conv0"}))
    assert len(nodes) == 1
    assert {n.name for n in nodes[0]} == {"conv0", "conv1"}

    nodes = list(pattern.match(graph, specify_node_names={"conv1"}))
    assert len(nodes) == 1
    assert {n.name for n in nodes[0]} == {"conv0", "conv1"}

    nodes = list(pattern.match(graph, specify_node_names={"conv2"}))
    assert len(nodes) == 0


def test_input_node_pattern_with_specify_node_names():
    graph = OnnxGraph(_build_test_graph2())
    pattern = InputNodePattern()
    nodes = list(pattern.match(graph, specify_node_names={"conv0"}))
    assert len(nodes) == 1
    assert nodes[0].name == "conv0"

    nodes = list(pattern.match(graph, specify_node_names={"conv1"}))
    assert len(nodes) == 0


def test_output_node_pattern_with_specify_node_names():
    graph = OnnxGraph(_build_test_graph2())
    pattern = OutputNodePattern()
    nodes = list(pattern.match(graph, specify_node_names={"add"}))
    assert len(nodes) == 1
    assert nodes[0].name == "add"

    nodes = list(pattern.match(graph, specify_node_names={"conv1"}))
    assert len(nodes) == 0
