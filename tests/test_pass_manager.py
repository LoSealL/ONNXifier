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

# pylint: disable=missing-function-docstring

import random

import onnx
import pytest

from onnxifier import OnnxGraph, PassManager
from onnxifier.passes import PASSES, Rewriter
from onnxifier.passes.pattern import SingleNodePattern
from onnxifier.utils import chdir


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


def test_pass_manager_include_and_exclude(tmp_path):
    passes = list(iter(PASSES))
    random.shuffle(passes)
    cut_pos = len(passes) // 2
    pass_manager = PassManager(passes[:cut_pos], passes[cut_pos:])
    graph = OnnxGraph(_empty_model())
    with chdir(tmp_path):
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


# ---------------------------------------------------------------------------
# _update_parent_graph: sync parent caller I/O after child function changes
# ---------------------------------------------------------------------------


def _model_with_function_io():
    """Model where main graph calls function Foo(x, unused) -> (y).

    Foo body: Add(x, unused) -> y   (uses both inputs)
    The *unused* input is not consumed downstream in the main graph,
    so an eliminate-unused-input style pass inside Foo could drop it.
    """
    # main graph: input x,unused -> Foo -> output y
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
    unused = onnx.helper.make_tensor_value_info("unused", onnx.TensorProto.FLOAT, [1])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])
    caller = onnx.helper.make_node(
        "Foo", ["x", "unused"], ["y"], name="foo_call", domain="foo"
    )
    graph = onnx.helper.make_graph([caller], "main", [x, unused], [y])

    foo_func = onnx.helper.make_function(
        "foo",
        "Foo",
        ["fx", "funused"],
        ["fy"],
        [onnx.helper.make_node("Add", ["fx", "funused"], ["fy"], name="add")],
        [onnx.helper.make_operatorsetid("foo", 1)],
    )

    model = onnx.helper.make_model(graph, functions=[foo_func])
    return model


def test_update_parent_graph_add_input():
    """When the child function gains a new input, the parent caller node
    should get the extra input appended and the parent graph should register
    it as a graph-level input."""
    m = _model_with_function_io()
    g = OnnxGraph(m)

    # A pass that adds a new input to the function body
    class _AddInputPass(Rewriter):
        def __init__(self):
            super().__init__(SingleNodePattern())

        def rewrite(self, graph, nodes):
            for node in nodes:
                if node.op_type == "Add":
                    new_in = "fnew"
                    node.input.append(new_in)
                    graph.set_value_info(
                        new_in, shape=[1], dtype=onnx.TensorProto.FLOAT
                    )
                    graph.set_input(node, new_in)
            return graph

    pm = PassManager([_AddInputPass()])
    result = pm.optimize(g, recursive=True, strict=True)

    # Caller node in the main graph should now have 3 inputs
    caller_found = False
    for n in result:
        caller = result.nodes[n]["pb"]
        if caller.op_type == "Foo":
            caller_found = True
            assert len(caller.input) == 3
            assert "fnew" in list(caller.input)
    assert caller_found


def test_update_parent_graph_add_output():
    """When the child function gains a new output, the caller node should
    grow its output list and the parent graph should register the new output."""
    m = _model_with_function_io()
    g = OnnxGraph(m)

    class _AddOutputPass(Rewriter):
        """Duplicate every Add node's output as a second output (test-only pass)."""

        def __init__(self):
            super().__init__(SingleNodePattern())

        def rewrite(self, graph, nodes):
            for node in nodes:
                if node.op_type == "Add":
                    new_out = node.output[0] + "_extra"
                    node.output.append(new_out)
                    graph.set_value_info(
                        new_out, shape=[1], dtype=onnx.TensorProto.FLOAT
                    )
                    graph.set_output(node, new_out)
            return graph

    pm = PassManager([_AddOutputPass()])
    result = pm.optimize(g, recursive=True, strict=True)

    caller_found = False
    for n in result:
        caller = result.nodes[n]["pb"]
        if caller.op_type == "Foo":
            caller_found = True
            assert len(caller.output) == 2
            assert "fy_extra" in list(caller.output)
    assert caller_found


def test_update_parent_graph_no_change():
    """When the child function I/O doesn't change, the caller node stays the same."""
    m = _model_with_function_io()
    g = OnnxGraph(m)

    # A pass that does nothing
    pm = PassManager([PassRecurse(debug_info={})])
    result = pm.optimize(g, recursive=True, strict=True)

    for n in result:
        caller = result.nodes[n]["pb"]
        if caller.op_type == "Foo":
            assert list(caller.input) == ["x", "unused"]
            assert list(caller.output) == ["y"]


# ---------------------------------------------------------------------------
# _assign_config_to_pass: edge cases
# ---------------------------------------------------------------------------


def test_config_non_dict_warning(caplog):
    """Config value that is not a dict should emit a warning."""
    PassManager(["function_in_test"], configs={"function_in_test": "bad_value"})
    assert "must be a dict" in caplog.text


def test_config_index_exceeds_boundary(caplog):
    """Index that exceeds available pass count should warn."""
    PassManager(
        ["function_in_test"],
        configs={"function_in_test:5": dict(args_x=1, args_y=1)},
    )
    assert "exceeds the boundary" in caplog.text


# ---------------------------------------------------------------------------
# Multi-user function: skipped during recursive optimization
# ---------------------------------------------------------------------------


def test_multi_user_function_skipped():
    """A function called by multiple nodes should be skipped (not optimised)."""
    m = _empty_model()
    m.graph.node.append(onnx.helper.make_node("Foo", [], [], name="foo1", domain="foo"))
    m.graph.node.append(onnx.helper.make_node("Foo", [], [], name="foo2", domain="foo"))
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

    debug_info = {}
    pm = PassManager([PassRecurse(debug_info)])
    pm.optimize(g, recursive=True, strict=True)

    # Both caller nodes are visited in the main graph
    assert "foo1" in debug_info
    assert "foo2" in debug_info
    # But the function body node should NOT be visited (multi-user → skipped)
    assert "const" not in debug_info


def test_print_all(capsys):
    PassManager.print_all()
    assert capsys.readouterr().out.strip()


def test_print_l1(capsys):
    PassManager.print_l1()
    assert capsys.readouterr().out.strip()


def test_print_l2(capsys):
    PassManager.print_l2()
    assert capsys.readouterr().out.strip()


def test_print_l3(capsys):
    PassManager.print_l3()
    assert capsys.readouterr().out.strip()


def test_print_specific(capsys):
    PassManager.print("function_in_test")
    assert "function_in_test" in capsys.readouterr().out


def test_repr():
    pm = PassManager(["function_in_test"])
    r = repr(pm)
    assert "function_in_test" in r
