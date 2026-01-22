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

# pylint: disable=missing-docstring

from typing import List

import numpy as np
import onnx
import pytest
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.onnx_pb import NodeProto

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET
from onnxifier.graph import OnnxGraph
from onnxifier.passes.pattern import GraphPattern, Pattern, SingleNodePattern
from onnxifier.passes.rewriter import Rewriter
from onnxifier.passes.utils import make_constant


class TestRewriter(Rewriter):
    def __init__(self, pattern, repeat, test_func):
        super().__init__(pattern, repeat)
        self.test_func = test_func

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        self.test_func(self, graph, nodes, *args, **kwargs)


def _make_graph(n_nodes=1):
    node = onnx.helper.make_node(
        "Dummy",
        ["x"],
        ["y"],
        "dummy",
        domain="test",
        attr1=1.0,
        attr2=2,
        attr3="3",
        attr4=onnx.numpy_helper.from_array(np.array([4])),
        attr6=onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [6, 6, 6]),
        attr7=[7.0, 7.0, 7.0],
        attr8=[8, 8, 8],
        attr9=["9", "9", "9"],
        attr10=[onnx.numpy_helper.from_array(np.array([10]))],
        attr12=[
            onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [12, 12, 12]),
            onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [12, 12, 12]),
        ],
    )
    nodes = [node]
    for i in range(1, n_nodes):
        nodes[-1].output[0] = f"n{i}"
        nodes.append(
            onnx.helper.make_node(
                "DummyTwo",
                [f"n{i}"],
                ["y"],
                f"dummy{i}",
                domain="test",
            )
        )
    graph = onnx.helper.make_graph(
        nodes,
        "test",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2, 3])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2, 3])],
    )
    return OnnxGraph(onnx.helper.make_model(graph))


def test_access_attribute():
    passed = False

    def _func(self, graph, nodes, *args, **kwargs):
        node = nodes[0]
        assert self.get_attribute(node, "attr1") == 1.0
        assert self.get_attribute(node, "attr2") == 2
        assert self.get_attribute(node, "attr3") == "3"
        assert (self.get_attribute(node, "attr4") == np.array([4])).all()
        assert isinstance(self.get_attribute(node, "attr6"), onnx.TypeProto)
        assert self.get_attribute(node, "attr7") == [7.0, 7.0, 7.0]
        assert self.get_attribute(node, "attr8") == [8, 8, 8]
        assert self.get_attribute(node, "attr9") == ["9", "9", "9"]
        assert (self.get_attribute(node, "attr10")[0] == np.array([10])).all()
        assert isinstance(self.get_attribute(node, "attr12")[0], onnx.TypeProto)
        assert isinstance(self.get_attribute(node, "attr12")[1], onnx.TypeProto)
        self.set_attribute(node, "attr1", 10.0)
        self.set_attribute(node, "attr2", 20)
        self.set_attribute(node, "attr3", "30")
        self.set_attribute(node, "attr4", np.array([40.0]))
        self.set_attribute(
            node,
            "attr6",
            onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [60, 60, 60]),
        )
        self.set_attribute(node, "attr7", [70.0, 70.0, 70.0])
        self.set_attribute(node, "attr8", [80, 80, 80])
        self.set_attribute(node, "attr9", ["90", "90", "90"])
        self.set_attribute(
            node, "attr10", [onnx.numpy_helper.from_array(np.array([100]))]
        )
        self.set_attribute(
            node,
            "attr12",
            [
                onnx.helper.make_tensor_type_proto(
                    onnx.TensorProto.FLOAT, [120, 120, 120]
                ),
                onnx.helper.make_tensor_type_proto(
                    onnx.TensorProto.FLOAT, [120, 120, 120]
                ),
            ],
        )
        nonlocal passed
        passed = True

    rewriter = TestRewriter(
        SingleNodePattern("Dummy").with_domain("test"), repeat=1, test_func=_func
    )
    rewriter(_make_graph())
    assert passed


def test_match_twice():
    repeats = 0

    def _func(self, graph, nodes, *args, **kwargs):
        assert len(nodes) == 2
        nonlocal repeats
        repeats += 1

    pattern = GraphPattern()
    pattern.add_edge(
        SingleNodePattern("Dummy").with_domain("test"),
        SingleNodePattern("DummyTwo").with_domain("test"),
    )
    target_repeats = np.random.randint(2, 100)
    rewriter = TestRewriter(pattern, repeat=target_repeats, test_func=_func)
    rewriter(_make_graph(2))
    assert repeats == target_repeats


def test_match_invalid_pattern():
    class InvalidPattern(Pattern):
        def match(self, graph: OnnxGraph, specify_node_names=None):
            for node in graph.nodes:
                return graph.nodes[node]["pb"]

    pattern = InvalidPattern()
    rewriter = TestRewriter(pattern, 1, lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError):
        rewriter(_make_graph())


def test_get_input_node():
    passed = False

    def _func(self, graph, nodes, *args, **kwargs):
        i1 = self.get_input_node(nodes[0], 0)
        i2 = self.get_input_node(nodes[0], -1)
        i3 = self.get_input_node(nodes[0], "n1")
        assert i1 is not None
        assert i2 is not None
        assert i1 is i2
        assert i2 is i3
        nonlocal passed
        passed = True

    rewriter = TestRewriter(SingleNodePattern("DummyTwo").with_domain("test"), 1, _func)
    rewriter(_make_graph(2))
    assert passed


def test_get_output_node():
    passed = False

    def _func(self, graph, nodes, *args, **kwargs):
        o1 = self.get_output_node(nodes[0], 0)[0]
        o2 = self.get_output_node(nodes[0], -1)[0]
        o3 = self.get_output_node(nodes[0], "n1")[0]
        assert o1 is not None
        assert o2 is not None
        assert o1 is o2
        assert o2 is o3
        nonlocal passed
        passed = True

    rewriter = TestRewriter(SingleNodePattern("Dummy").with_domain("test"), 1, _func)
    rewriter(_make_graph(3))
    assert passed


def test_get_value():
    cst = make_constant("cst", np.array([1], np.int64))
    add = make_node("Add", ["x", cst.output[0]], ["y"])
    shapeof = make_node("Shape", [cst.output[0]], ["z"])
    mul = make_node("Mul", ["y", "z"], ["w"])
    model = make_model(
        make_graph(
            [cst, add, shapeof, mul],
            "test",
            [],
            [make_tensor_value_info("w", onnx.TensorProto.INT64, [1])],
            [onnx.numpy_helper.from_array(np.array([2], np.int64), "x")],
        ),
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    graph = OnnxGraph(model)

    def _func(self, graph, nodes, *args, **kwargs):
        node = nodes[0]
        if node.op_type == "Add":
            cst = self.get_input_node(node, 1)
            assert (self.get_value(cst) == np.array([1], np.int64)).all()
            assert (self.get_value(node.input[1]) == np.array([1], np.int64)).all()
            assert (self.get_value("x") == np.array([2], np.int64)).all()
        else:
            assert node.op_type == "Mul"
            shapeof = self.get_input_node(node, 1)
            assert (self.get_value(shapeof) == np.array([1], np.int64)).all()
            result = self.get_value(node.input[0])
            assert np.all(result == np.array([3], np.int64))

    pattern = SingleNodePattern("Add") | SingleNodePattern("Mul")
    rewriter = TestRewriter(pattern, 1, _func)
    rewriter(graph)


def test_get_constant_values():
    cst_i = make_node("Constant", [], ["int"], name="const_int", value_int=1)
    cst_ints = make_node(
        "Constant", [], ["ints"], name="const_ints", value_ints=[1, 2, 3]
    )
    cst_f = make_node("Constant", [], ["float"], name="const_float", value_float=1.0)
    cst_floats = make_node(
        "Constant", [], ["floats"], name="const_floats", value_floats=[1.0, 2.0, 3.0]
    )
    cst_s = make_node("Constant", [], ["string"], name="const_string", value_string="1")
    cst_strings = make_node(
        "Constant", [], ["strings"], name="const_strings", value_strings=["1", "2", "3"]
    )
    model = make_model(
        make_graph(
            [cst_i, cst_ints, cst_f, cst_floats, cst_s, cst_strings],
            "test_cst",
            [],
            [
                make_tensor_value_info("int", onnx.TensorProto.INT64, []),
                make_tensor_value_info("ints", onnx.TensorProto.INT64, [3]),
                make_tensor_value_info("float", onnx.TensorProto.FLOAT, []),
                make_tensor_value_info("floats", onnx.TensorProto.FLOAT, [3]),
                make_tensor_value_info("string", onnx.TensorProto.STRING, []),
                make_tensor_value_info("strings", onnx.TensorProto.STRING, [3]),
            ],
        ),
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    graph = OnnxGraph(model)

    def _func(self, graph, nodes, *args, **kwargs):
        node = nodes[0]
        v = self.get_value_or_die(node.output[0])
        if node.name == "const_int":
            assert v.dtype == np.int64
            assert v == 1
        elif node.name == "const_ints":
            assert v.dtype == np.int64
            assert (v == np.array([1, 2, 3])).all()
        elif node.name == "const_float":
            assert v.dtype == np.float32
            assert v == 1.0
        elif node.name == "const_floats":
            assert v.dtype == np.float32
            assert (v == np.array([1.0, 2.0, 3.0])).all()
        elif node.name == "const_string":
            assert v.dtype == np.dtype("S1")
            assert v.item() == b"1"
        elif node.name == "const_strings":
            assert v.dtype == np.dtype("S1")
            assert (v == np.array([b"1", b"2", b"3"])).all()

    pattern = SingleNodePattern("Constant")
    rewriter = TestRewriter(pattern, 1, _func)
    rewriter(graph)
