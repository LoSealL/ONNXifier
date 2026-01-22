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

import random

import onnx
import pytest

from onnxifier import OnnxGraph, PassManager
from onnxifier.passes import PASSES, Rewriter
from onnxifier.passes.pattern import SingleNodePattern


@PASSES.register(name="function_in_test")
def fake_pass_function(graph, args_x, args_y):
    assert args_x == args_y
    return graph


@PASSES.register(name="class_in_test")
class FakeClassPass(Rewriter):
    """Fake Pass"""

    def __init__(self):
        super().__init__(SingleNodePattern())

    def rewrite(self, graph, nodes, args_x=0, args_y=2):
        assert args_x == args_y


@PASSES.register("cycle_a", deps=["cycle_b"])
def fake_cycle_a(graph):
    return graph


@PASSES.register("cycle_b", deps=["cycle_a"])
def fake_cycle_b(graph):
    return graph


@PASSES.register("patch_a", patch=["patch_b"])
def fake_patch_a(graph):
    return graph


@PASSES.register("patch_b", patch=["patch_a"])
def fake_patch_b(graph):
    return graph


def _empty_model():
    return onnx.helper.make_model(
        graph=onnx.helper.make_graph([], "empty", [], []),
    )


def test_pass_manager_default():
    graph = OnnxGraph(_empty_model())
    pass_manager = PassManager()
    pass_manager.optimize(graph)


def test_pass_manager_include_and_exclude():
    passes = list(iter(PASSES))
    random.shuffle(passes)
    cut_pos = len(passes) // 2
    pass_manager = PassManager(passes[:cut_pos], passes[cut_pos:])
    graph = OnnxGraph(_empty_model())
    pass_manager.optimize(graph)


def test_pass_manager_include_instance():
    pass_manager = PassManager([FakeClassPass()])
    graph = OnnxGraph(_empty_model())
    pass_manager.optimize(graph)


def test_pass_manager_include_warnings(caplog):
    PassManager(["not_exist_pass"])
    assert "WARNING" in caplog.text


def test_pass_manager_with_configs():
    pass_manager = PassManager(
        ["function_in_test", "class_in_test", "class_in_test", "function_in_test"],
        configs={
            "function_in_test": dict(args_x=1, args_y=1),
            "class_in_test:0": dict(args_x=1, args_y=1),
            "class_in_test:1": dict(args_x="2", args_y="2"),
        },
    )
    graph = OnnxGraph(_empty_model())
    pass_manager.optimize(graph, strict=True)


def test_pass_manager_check_cycle():
    pm = PassManager(["cycle_a", "cycle_b"])
    with pytest.raises(RuntimeError):
        pm.optimize(OnnxGraph(_empty_model()), strict=True)

    pm = PassManager(["patch_a", "patch_b"])
    with pytest.raises(RuntimeError):
        pm.optimize(OnnxGraph(_empty_model()), strict=True)


def test_pass_child():
    assert "patch_b" in PASSES.child("patch_b")
    assert "patch_a" in PASSES.child(["patch_a", "patch_b"])
    assert "cycle_a" not in PASSES.child(["patch_a", "patch_b"])

    with pytest.raises(KeyError):
        PASSES.child("not_exist")


@PASSES.register("test_pass_recurse")
class PassRecurse(Rewriter):
    """Fake Pass"""

    def __init__(self, debug_info=None):
        super().__init__(SingleNodePattern())
        self.debug_info = debug_info

    def rewrite(self, graph, nodes):
        self.debug_info[nodes[0].name] = nodes[0].op_type
        return graph


def test_optimize_recursively():
    m = _empty_model()
    m.graph.node.append(onnx.helper.make_node("Foo", [], [], name="foo", domain="foo"))
    m.functions.append(
        onnx.helper.make_function(
            "foo",
            "Foo",
            [],
            [],
            [onnx.helper.make_node("Const", [], ["out"], name="const")],
            [onnx.helper.make_operatorsetid("foo", 1)],
        )
    )
    g = OnnxGraph(m)

    debug = {}
    rewriter = PassRecurse(debug)
    pm = PassManager([rewriter])
    pm.optimize(g, recursive=False)

    assert debug["foo"] == "Foo"
    assert "const" not in debug

    pm.optimize(g, recursive=True)
    assert debug["const"] == "Const"


def test_optimize_recursively_with_nested_functions_out_of_order():
    m = _empty_model()
    m.graph.node.append(onnx.helper.make_node("Foo", [], [], name="foo", domain="foo"))
    m.functions.append(
        onnx.helper.make_function(
            "foo",
            "Foo",
            [],
            [],
            [onnx.helper.make_node("Bar", [], ["out"], name="bar_call")],
            [onnx.helper.make_operatorsetid("foo", 1)],
        )
    )
    m.functions.append(
        onnx.helper.make_function(
            "bar",
            "Bar",
            [],
            [],
            [onnx.helper.make_node("Const", [], ["out"], name="const")],
            [onnx.helper.make_operatorsetid("bar", 1)],
        )
    )
    g = OnnxGraph(m)

    debug = {}
    rewriter = PassRecurse(debug)
    pm = PassManager([rewriter])
    pm.optimize(g, recursive=True, strict=True)

    assert debug["foo"] == "Foo"
    assert debug["bar_call"] == "Bar"
    assert debug["const"] == "Const"


def test_optimize_with_specify_node_names():
    m = _empty_model()
    m.graph.node.append(onnx.helper.make_node("Foo", [], [], name="foo", domain="foo"))
    m.graph.node.append(onnx.helper.make_node("Bar", [], [], name="bar", domain="bar"))
    g = OnnxGraph(m)

    pm = PassManager(
        ["function_in_test"], configs={"function_in_test": {"args_x": 1, "args_y": 2}}
    )
    # This function_in_test should fail since args_x != args_y
    with pytest.raises(AssertionError):
        pm.optimize(g, strict=True, specify_node_names={"foo"})
    with pytest.raises(AssertionError):
        # specify_node_names has no effect to function passes
        pm.optimize(g, strict=True, specify_node_names={"baz"})

    pm = PassManager(
        ["class_in_test"], configs={"class_in_test": {"args_x": 2, "args_y": 3}}
    )
    # This class_in_test should fail since args_x != args_y
    with pytest.raises(AssertionError):
        pm.optimize(g, strict=True, specify_node_names={"foo"})
    # However, specify node name = "baz" matches no nodes
    pm.optimize(g, strict=True, specify_node_names={"baz"})
