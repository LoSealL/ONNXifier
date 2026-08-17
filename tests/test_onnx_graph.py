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

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnx.external_data_helper
import onnx.helper as oh
import pooch
import pytest

from onnxifier import ONNXIFIER_OPSET, OnnxGraph, convert_graph
from onnxifier.utils import chdir


def _pick_random_nodes(graph: OnnxGraph, num_nodes: int = 1):
    indices = list(range(len(graph)))
    random.shuffle(indices)
    node_names = np.array(graph)[indices[:num_nodes]].tolist()
    if len(node_names) == 1:
        return node_names[0]
    return node_names


def test_loading_models(classification_models):
    model_url, hash_value = classification_models
    model = onnx.load_model(pooch.retrieve(model_url, hash_value))
    OnnxGraph(model)


def test_add_remove_node(classification_models):
    model_url, hash_value = classification_models
    model = onnx.load_model(pooch.retrieve(model_url, hash_value))

    graph = OnnxGraph(model)
    node_name = _pick_random_nodes(graph)
    node = graph.nodes[node_name]["pb"]
    graph.remove_onnx_node(node, no_replace=True)
    onnx.checker.check_model(graph.model)

    graph.add_onnx_node(node)
    onnx.checker.check_model(graph.model)

    graph.remove_onnx_node(node.name)
    try:
        onnx.checker.check_model(graph.model)
    except onnx.checker.ValidationError:
        pass
    else:
        raise RuntimeError("onnx.checker.ValidationError is expected to raise")

    graph.add_onnx_node(node)
    onnx.checker.check_model(graph.model)


def test_make_subgraph(classification_models):
    model_url, hash_value = classification_models
    model = onnx.load_model(pooch.retrieve(model_url, hash_value))

    graph = convert_graph(model, print_passes=False)
    model = onnx.version_converter.convert_version(graph.model, 19)
    graph = OnnxGraph(model)

    sub_node_names = _pick_random_nodes(graph, len(graph) // 5)
    sub_nodes = [graph.nodes[i]["pb"] for i in sub_node_names]

    sub_from_nodes = graph.onnx_subgraph(sub_nodes)
    sub_from_names = graph.onnx_subgraph(sub_node_names)
    onnx.checker.check_model(sub_from_nodes.model)
    onnx.checker.check_model(sub_from_names.model)


def test_tensor_info(classification_models):
    model_url, hash_value = classification_models
    model = onnx.load_model(pooch.retrieve(model_url, hash_value))

    graph = OnnxGraph(model)
    node_name = _pick_random_nodes(graph)
    node = graph.nodes[node_name]["pb"]
    for input_name in node.input:
        graph.tensor_info(input_name)
    for output_name in node.output:
        graph.tensor_info(output_name)


def test_model_save(classification_models):
    model_url, hash_value = classification_models
    model = onnx.load_model(pooch.retrieve(model_url, hash_value))

    graph = OnnxGraph(model)
    with tempfile.TemporaryDirectory() as tmpdir:
        graph.save(tmpdir + "/model")
        # text format is too slow
        # graph.save(tmpdir + "/model", format="textproto")
        # graph.save(tmpdir + "/model", format="json")
        # graph.save(tmpdir + "/model", format="onnxtxt")


def test_subgraph_outputs():
    def _make_test_model():
        """a  (b)
          |   /
          (Add)
          / |
        (x) y
        """
        node1 = onnx.helper.make_node(
            "Constant",
            [],
            ["a"],
            name="A",
            value=onnx.numpy_helper.from_array(np.zeros([16], "float32")),
        )
        add = onnx.helper.make_node("Add", ["a", "b"], ["c"], name="Add")
        id1 = onnx.helper.make_node("Identity", ["c"], ["x"], "ID1")
        id2 = onnx.helper.make_node("Identity", ["c"], ["y"], "ID2")
        graph = onnx.helper.make_graph(
            [node1, add, id1, id2],
            "test",
            [],
            [
                onnx.helper.make_value_info(
                    "x", onnx.helper.make_tensor_type_proto(1, [16])
                ),
                onnx.helper.make_value_info(
                    "y", onnx.helper.make_tensor_type_proto(1, [16])
                ),
            ],
            [onnx.numpy_helper.from_array(np.zeros([1], "float32"), "b")],
        )
        return onnx.helper.make_model(graph, opset_imports=[ONNXIFIER_OPSET])

    graph = OnnxGraph(_make_test_model())
    subgraph = graph.onnx_subgraph(["Add", "ID1"])
    assert len(subgraph) == 2
    assert len(subgraph.inputs) == 1
    assert len(subgraph.outputs) == 2


def test_subgraph_outputs_case2():
    def _make_test_model():
        """a   b
          |   /
          (Add)
          / |
        (x) c
        """
        add = onnx.helper.make_node("Add", ["a", "b"], ["c"], name="Add")
        id1 = onnx.helper.make_node("Identity", ["c"], ["x"], "ID1")
        graph = onnx.helper.make_graph(
            [add, id1],
            "test",
            [
                onnx.helper.make_value_info(
                    "a", onnx.helper.make_tensor_type_proto(1, [16])
                ),
                onnx.helper.make_value_info(
                    "b", onnx.helper.make_tensor_type_proto(1, [16])
                ),
            ],
            [
                onnx.helper.make_value_info(
                    "x", onnx.helper.make_tensor_type_proto(1, [16])
                ),
                onnx.helper.make_value_info(
                    "c", onnx.helper.make_tensor_type_proto(1, [16])
                ),
            ],
        )
        return onnx.helper.make_model(graph, opset_imports=[ONNXIFIER_OPSET])

    graph = OnnxGraph(_make_test_model())
    subgraph = graph.onnx_subgraph(["Add", "ID1"])
    assert len(subgraph) == 2
    assert len(subgraph.inputs) == 2
    assert len(subgraph.outputs) == 2


def test_onnx_node_hashable():
    node0 = onnx.helper.make_node("Add", ["x", "y"], ["z"], name="add0")
    node1 = onnx.helper.make_node("Add", ["x", "y"], ["z"], name="add1")
    node2 = onnx.helper.make_node("Sub", ["x", "y"], ["z"], name="add0")
    assert hash(node0) != hash(node1)
    # NOTE: hash value of onnx NodeProto is only determined by its name,
    #       type or attributes are not included.
    assert hash(node0) == hash(node2)


def test_add_duplicated_node():
    add0 = onnx.helper.make_node("Add", ["x", "y"], ["z"], name="add0")
    relu = onnx.helper.make_node("Relu", ["z"], ["w"], name="relu")
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [add0, relu],
            "test_graph",
            [
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [32]),
                onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [32]),
            ],
            [onnx.helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, [32])],
        )
    )
    graph = OnnxGraph(model)
    # replace add0 with Sub
    new_add0 = onnx.helper.make_node("Sub", ["x", "y"], ["z"], name="add0")
    graph.add_onnx_node(new_add0)
    assert graph.nodes["add0"]["pb"].op_type == "Sub"


def test_add_node_override_outputs():
    add0 = onnx.helper.make_node("Add", ["x", "y"], ["z"], name="add0")
    act0 = onnx.helper.make_node("Relu", ["z"], ["w"], name="relu")
    act1 = onnx.helper.make_node("Sin", ["w"], ["u"], name="sin")
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [add0, act0, act1],
            "test",
            [
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [32]),
                onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [32]),
            ],
            [
                onnx.helper.make_tensor_value_info("u", onnx.TensorProto.FLOAT, [32]),
            ],
        )
    )
    graph = OnnxGraph(model)
    new_add0 = onnx.helper.make_node("Add", ["x", "y"], ["w"], name="add0")
    graph.add_onnx_node(new_add0)
    graph.remove_onnx_node("relu")
    onnx.checker.check_model(graph.model, True)


def test_graph_property():
    g = onnx.helper.make_graph([], "test", [], [])
    m = onnx.helper.make_model(g, opset_imports=[ONNXIFIER_OPSET])
    graph = OnnxGraph(m)
    assert graph.name == "test"
    assert graph.ir_version == onnx.IR_VERSION
    assert graph.opset_version == ONNXIFIER_OPSET.version

    graph.opset_version = 10
    assert graph.opset_version == 10


def test_graph_save_as_external(tmp_path):
    # pylint: disable=protected-access
    g = onnx.helper.make_graph(
        nodes=[
            onnx.helper.make_node(
                "Constant",
                [],
                ["att_big"],
                value=onnx.numpy_helper.from_array(
                    np.ones([1024], dtype=np.float32), "att_big"
                ),
            ),
            onnx.helper.make_node(
                "Constant",
                [],
                ["att_small"],
                value=onnx.numpy_helper.from_array(
                    np.ones([1023], dtype=np.int8), "att_small"
                ),
            ),
        ],
        name="test",
        inputs=[],
        outputs=[],
        initializer=[
            onnx.numpy_helper.from_array(np.ones([1024], dtype=np.float32), "big"),
            onnx.numpy_helper.from_array(np.ones([1023], dtype=np.int8), "small"),
        ],
    )
    f = onnx.helper.make_function(
        "test",
        "foo",
        [],
        ["b"],
        [
            onnx.helper.make_node(
                "Constant",
                [],
                ["b"],
                value=onnx.numpy_helper.from_array(
                    np.ones([1024], dtype=np.float32), "b"
                ),
            )
        ],
        [ONNXIFIER_OPSET],
        ["attr_big"],
        [
            onnx.helper.make_attribute(
                "attr_big",
                onnx.numpy_helper.from_array(
                    np.ones([1024], dtype=np.float32), "attr_big"
                ),
            )
        ],
    )
    model = onnx.helper.make_model(g, opset_imports=[ONNXIFIER_OPSET], functions=[f])
    onnx.checker.check_model(model, True, True)
    graph = OnnxGraph(model)
    graph.restore_tensors_from_external()

    graph.save(tmp_path / "model.onnx", save_as_external_data=True)
    model = onnx.load_model(tmp_path / "model.onnx", load_external_data=False)
    external_tensors = [
        t
        for t in onnx.external_data_helper._get_all_tensors(model)
        if onnx.external_data_helper.uses_external_data(t)
    ]
    # onnx enumerating tensors could be duplicated, because it yield function twice
    onnx.TensorProto.__hash__ = lambda x: hash(id(x))  # type: ignore
    # function attribute could be saved as external data
    assert len(set(external_tensors)) == 3

    with pytest.raises(FileNotFoundError):
        OnnxGraph(model, base_dir="wrong-base")
    with chdir(tmp_path):
        ext_graph = OnnxGraph(model)
    ext_data = list(ext_graph.external_data)
    assert len(ext_data) == 1
    assert ext_data[0] == Path(tmp_path) / "model.onnx.data"
    ext_graph.restore_tensors_from_external()
    external_tensors = [
        t
        for t in onnx.external_data_helper._get_all_tensors(ext_graph.model)
        if onnx.external_data_helper.uses_external_data(t)
    ]
    assert len(external_tensors) == 0
    # save to a new location because graph has been restored
    ext_graph.save(tmp_path / "model.onnx", save_as_external_data=True)


def test_external_base_resolved_absolute(tmp_path):
    """A relative base must resolve to absolute at set time, so data
    resolution keeps working after the process chdirs elsewhere."""
    g = onnx.helper.make_graph(
        nodes=[],
        name="test",
        inputs=[],
        outputs=[],
        initializer=[
            onnx.numpy_helper.from_array(np.ones([1024], dtype=np.float32), "big"),
        ],
    )
    model = onnx.helper.make_model(g, opset_imports=[ONNXIFIER_OPSET])
    (tmp_path / "sub").mkdir()
    with chdir(tmp_path / "sub"):
        graph = OnnxGraph(model)
        graph.save("model.onnx", save_as_external_data=True)
        rel_base = os.path.relpath(tmp_path / "sub")
        loaded = OnnxGraph(
            onnx.load_model("model.onnx", load_external_data=False), base_dir=rel_base
        )
    assert Path(loaded.external_base).is_absolute()
    # from a different cwd the data must still resolve and restore
    with chdir(tmp_path):
        loaded.restore_tensors_from_external()
    big = onnx.numpy_helper.to_array(loaded.model.graph.initializer[0])
    assert big.shape == (1024,)


def test_convert_tensors_rebase(tmp_path):
    """Saving to an absolute location outside the current base re-bases
    instead of raising, and the stored location stays relative."""
    g = onnx.helper.make_graph(
        nodes=[],
        name="test",
        inputs=[],
        outputs=[],
        initializer=[
            onnx.numpy_helper.from_array(np.ones([1024], dtype=np.float32), "big"),
        ],
    )
    model = onnx.helper.make_model(g, opset_imports=[ONNXIFIER_OPSET])
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    with chdir(src):
        graph = OnnxGraph(model)
        graph.save("model.onnx", save_as_external_data=True)
        # old code raised ValueError here
        graph.save(dst / "model.onnx", save_as_external_data=True)
    reloaded = onnx.load_model(dst / "model.onnx", load_external_data=False)
    for t in onnx.external_data_helper._get_all_tensors(reloaded):
        if onnx.external_data_helper.uses_external_data(t):
            loc = onnx.external_data_helper.ExternalDataInfo(t).location
            assert not Path(loc).is_absolute()


def test_onnx_subgraph_filters_unreachable_functions():
    """Sub-models carry only functions the subgraph calls (transitively)."""
    nodes = [
        oh.make_node(
            "Constant",
            [],
            ["c"],
            value=onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), "c"),
            name="C0",
        ),
        oh.make_node("Shape", ["X"], ["s"], name="S0"),
        oh.make_node("Gather", ["s", "c"], ["g"], name="G0"),
        oh.make_node("MyFn", ["g"], ["y"], domain="hyper", name="call0"),
        oh.make_node("NestedFn", ["y"], ["z"], domain="hyper", name="call1"),
    ]
    graph = oh.make_graph(
        nodes,
        "t",
        [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 2])],
        [oh.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [2, 2])],
    )
    opsets = [oh.make_operatorsetid("", 23), oh.make_operatorsetid("hyper", 1)]
    # unrelated function with an op the reference evaluator cannot implement
    gelu = oh.make_function(
        "hyper",
        "UnrelatedGelu",
        ["a"],
        ["b"],
        [oh.make_node("Gelu", ["a"], ["b"], name="fn/gelu")],
        opsets,
    )
    my_fn = oh.make_function(
        "hyper",
        "MyFn",
        ["a"],
        ["b"],
        [oh.make_node("Neg", ["a"], ["b"], name="fn/neg")],
        opsets,
    )
    nested = oh.make_function(
        "hyper",
        "NestedFn",
        ["a"],
        ["b"],
        [oh.make_node("MyFn", ["a"], ["b"], domain="hyper", name="fn/call")],
        opsets,
    )
    model = oh.make_model(graph, opset_imports=opsets, functions=[gelu, my_fn, nested])
    g = OnnxGraph(model)

    # subgraph without the calls: no function attached
    sub = g.onnx_subgraph(["C0", "S0", "G0"])
    assert set(sub.functions) == set()

    # subgraph with the calls: callees attached transitively, unrelated not
    sub = g.onnx_subgraph(["C0", "S0", "G0", "call0", "call1"])
    assert set(sub.functions) == {"MyFn", "NestedFn"}


def test_model_orders_subgraph_outer_scope_refs():
    """Serialization must emit outer-scope producers before the If that
    references them from a branch, even without an explicit edge."""
    then_g = oh.make_graph(
        [oh.make_node("Neg", ["x_out"], ["y"], name="then/neg")],
        "then",
        [],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
    )
    else_g = oh.make_graph(
        [oh.make_node("Identity", ["x_out"], ["y"], name="else/id")],
        "else",
        [],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
    )
    nodes = [
        oh.make_node(
            "If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g, name="if0"
        ),
        # only the If's branch references this producer
        oh.make_node("Mul", ["a", "b"], ["x_out"], name="mul0"),
    ]
    graph = oh.make_graph(
        nodes,
        "t",
        [
            oh.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [2]),
            oh.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [2]),
            oh.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [1]),
        ],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
    )
    model = oh.make_model(graph, opset_imports=[ONNXIFIER_OPSET])
    g = OnnxGraph(model)
    edges_before = g.number_of_edges()
    serialized = g.model
    names = [n.name for n in serialized.graph.node]
    assert names.index("mul0") < names.index("if0")
    onnx.checker.check_model(serialized)
    # implicit edges must not leak into the graph
    assert g.number_of_edges() == edges_before


def test_graph_sibling_nodes():
    """Test get sibling nodes."""

    nodes = [
        oh.make_node("Identity", ["x"], ["y"], name="id1"),
        oh.make_node("Sin", ["y"], ["z"], name="sin1"),
        oh.make_node("Cos", ["y"], ["w"], name="cos1"),
    ]
    model = oh.make_model(
        oh.make_graph(
            nodes,
            "sibling_graph_test",
            inputs=[oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
            outputs=[
                oh.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [1]),
                oh.make_tensor_value_info("w", onnx.TensorProto.FLOAT, [1]),
            ],
        ),
        ir_version=onnx.IR_VERSION,
        opset_imports=[oh.make_operatorsetid("", 21)],
    )
    graph = OnnxGraph(model)
    sib1 = graph.onnx_siblings("sin1")
    sib2 = graph.onnx_siblings("cos1")
    assert len(sib1) == 1
    assert len(sib2) == 1
    assert sib1[0].name == "cos1"
    assert sib2[0].name == "sin1"


def test_set_value_info_for_input():
    """Test that set_value_info updates graph input when name is in inputs."""
    nodes = [oh.make_node("Identity", ["x"], ["y"], name="id1")]
    model = oh.make_model(
        oh.make_graph(
            nodes,
            "test_graph",
            inputs=[oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [10])],
            outputs=[oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [10])],
        ),
        ir_version=onnx.IR_VERSION,
        opset_imports=[oh.make_operatorsetid("", 23)],
    )
    graph = OnnxGraph(model)

    # Update value info for input 'x' with new shape
    graph.set_value_info("x", shape=[20], dtype=onnx.TensorProto.FLOAT)

    # Verify the input value info was updated
    assert "x" in graph.inputs
    input_info = graph.input[graph.inputs["x"]]
    shape = input_info.type.tensor_type.shape
    assert shape.dim[0].dim_value == 20


def test_set_value_info_for_output():
    """Test that set_value_info updates graph output when name is in outputs."""
    nodes = [oh.make_node("Identity", ["x"], ["y"], name="id1")]
    model = oh.make_model(
        oh.make_graph(
            nodes,
            "test_graph",
            inputs=[oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [10])],
            outputs=[oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [10])],
        ),
        ir_version=onnx.IR_VERSION,
        opset_imports=[oh.make_operatorsetid("", 21)],
    )
    graph = OnnxGraph(model)

    # Update value info for output 'y' with new shape
    graph.set_value_info("y", shape=[20], dtype=onnx.TensorProto.FLOAT)

    # Verify the output value info was updated
    assert "y" in graph.outputs
    output_info = graph.output[graph.outputs["y"]]
    shape = output_info.type.tensor_type.shape
    assert shape.dim[0].dim_value == 20
