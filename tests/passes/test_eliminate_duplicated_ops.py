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

from unittest.mock import patch

from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier import ONNXIFIER_OPSET, PassManager
from onnxifier.graph import OnnxGraph
from onnxifier.passes.fusion.eliminate_duplicated_ops import (
    EliminateDuplicatedTranspose,
)


def _optimize(model):
    graph_opt = OnnxGraph(model)
    pm = PassManager(["eliminate_duplicated_transpose"])
    return pm.optimize(graph_opt, strict=True)


def _build_basic_chain_model():
    shape_const = make_node(
        "Constant",
        [],
        ["shape_val"],
        value=make_tensor("shape_tensor", TensorProto.INT64, [3], [1, 2, 3]),
    )
    t1 = make_node("Transpose", ["x"], ["t1_out"], perm=[0, 2, 1])
    r = make_node("Reshape", ["t1_out", "shape_val"], ["r_out"])
    t2 = make_node("Transpose", ["r_out"], ["y"], perm=[0, 2, 1])
    graph = make_graph(
        [shape_const, t1, r, t2],
        "graph",
        [make_value_info("x", make_tensor_type_proto(TensorProto.FLOAT, [1, 3, 2]))],
        [make_value_info("y", make_tensor_type_proto(TensorProto.FLOAT, [1, 3, 2]))],
        [],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def test_eliminate_duplicated_transpose():
    reshape_const = make_node(
        "Constant",
        [],
        ["shape_val"],
        value=make_tensor("shape_tensor", TensorProto.INT64, [3], [1, 2, 3]),
    )

    t1 = make_node("Transpose", ["x"], ["t1_out"], perm=[0, 2, 1])
    r = make_node("Reshape", ["t1_out", "shape_val"], ["r_out"])
    t2 = make_node("Transpose", ["r_out"], ["y"], perm=[0, 2, 1])

    graph = make_graph(
        [reshape_const, t1, r, t2],
        "graph",
        [make_value_info("x", make_tensor_type_proto(TensorProto.FLOAT, [1, 3, 2]))],
        [make_value_info("y", make_tensor_type_proto(TensorProto.FLOAT, [1, 3, 2]))],
        [],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])

    graph_opt = _optimize(model)

    # Should remove the redundant transpose pair.
    op_types = [n.op_type for n in graph_opt.model.graph.node]
    assert "Transpose" not in op_types

    check_model(graph_opt.model, True)


def test_skip_when_not_monopath():
    t1 = make_node("Transpose", ["x"], ["t1_out"], perm=[0, 2, 1])
    relu = make_node("Relu", ["t1_out"], ["y"])
    sin = make_node("Sin", ["t1_out"], ["z"])
    graph = make_graph(
        [t1, relu, sin],
        "graph",
        [make_value_info("x", make_tensor_type_proto(TensorProto.FLOAT, [1, 3, 2]))],
        [
            make_value_info("y", make_tensor_type_proto(TensorProto.FLOAT, [1, 2, 3])),
            make_value_info("z", make_tensor_type_proto(TensorProto.FLOAT, [1, 2, 3])),
        ],
        [],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])

    graph_opt = _optimize(model)
    op_types = [n.op_type for n in graph_opt.model.graph.node]
    assert op_types.count("Transpose") == 1


def test_skip_when_tail_is_not_transpose_or_reshape():
    t1 = make_node("Transpose", ["x"], ["t1_out"], perm=[0, 2, 1])
    unsq = make_node("Unsqueeze", ["t1_out", "axes"], ["y"])
    graph = make_graph(
        [t1, unsq],
        "graph",
        [make_value_info("x", make_tensor_type_proto(TensorProto.FLOAT, [1, 3, 2]))],
        [make_value_info("y", make_tensor_type_proto(TensorProto.FLOAT, [1, 1, 2, 3]))],
        [make_tensor("axes_t", TensorProto.INT64, [1], [1])],
    )
    # unsqueeze takes axes from initializer named "axes"
    graph.node[1].input[1] = "axes_t"
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])

    graph_opt = _optimize(model)
    op_types = [n.op_type for n in graph_opt.model.graph.node]
    assert "Unsqueeze" in op_types
    assert "Transpose" in op_types


def test_skip_when_shape_is_dynamic():
    t1 = make_node("Transpose", ["x"], ["t1_out"], perm=[0, 2, 1])
    r = make_node("Reshape", ["t1_out", "shape_val"], ["r_out"])
    t2 = make_node("Transpose", ["r_out"], ["y"], perm=[0, 2, 1])
    shape_const = make_node(
        "Constant",
        [],
        ["shape_val"],
        value=make_tensor("shape_tensor", TensorProto.INT64, [3], [1, 2, 3]),
    )
    graph = make_graph(
        [shape_const, t1, r, t2],
        "graph",
        [make_value_info("x", make_tensor_type_proto(TensorProto.FLOAT, ["N", 3, 2]))],
        [make_value_info("y", make_tensor_type_proto(TensorProto.FLOAT, ["N", 3, 2]))],
        [],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])

    graph_opt = _optimize(model)
    op_types = [n.op_type for n in graph_opt.model.graph.node]
    assert op_types.count("Transpose") == 2


def test_skip_when_extra_input_value_unavailable():
    t1 = make_node("Transpose", ["x"], ["t1_out"], perm=[0, 2, 1])
    r = make_node("Reshape", ["t1_out", "shape_in"], ["r_out"])
    t2 = make_node("Transpose", ["r_out"], ["y"], perm=[0, 2, 1])
    graph = make_graph(
        [t1, r, t2],
        "graph",
        [
            make_value_info("x", make_tensor_type_proto(TensorProto.FLOAT, [1, 3, 2])),
            make_value_info("shape_in", make_tensor_type_proto(TensorProto.INT64, [3])),
        ],
        [make_value_info("y", make_tensor_type_proto(TensorProto.FLOAT, [1, 3, 2]))],
        [],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])

    graph_opt = _optimize(model)
    op_types = [n.op_type for n in graph_opt.model.graph.node]
    assert op_types.count("Transpose") == 2


def test_skip_when_layout_not_cancelled():
    shape_const = make_node(
        "Constant",
        [],
        ["shape_val"],
        value=make_tensor("shape_tensor", TensorProto.INT64, [3], [2, 4, 3]),
    )
    t1 = make_node("Transpose", ["x"], ["t1_out"], perm=[0, 2, 1])
    r = make_node("Reshape", ["t1_out", "shape_val"], ["r_out"])
    t2 = make_node("Transpose", ["r_out"], ["y"], perm=[2, 1, 0])
    graph = make_graph(
        [shape_const, t1, r, t2],
        "graph",
        [make_value_info("x", make_tensor_type_proto(TensorProto.FLOAT, [2, 3, 4]))],
        [make_value_info("y", make_tensor_type_proto(TensorProto.FLOAT, [3, 4, 2]))],
        [],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])

    graph_opt = _optimize(model)
    op_types = [n.op_type for n in graph_opt.model.graph.node]
    assert op_types.count("Transpose") == 2


def test_skip_on_evaluator_exception():
    # Force evaluator failure with an invalid reshape target size.
    shape_const = make_node(
        "Constant",
        [],
        ["shape_val"],
        value=make_tensor("shape_tensor", TensorProto.INT64, [1], [5]),
    )
    t1 = make_node("Transpose", ["x"], ["t1_out"], perm=[0, 2, 1])
    r = make_node("Reshape", ["t1_out", "shape_val"], ["r_out"])
    t2 = make_node("Transpose", ["r_out"], ["y"], perm=[0, 2, 1])
    graph = make_graph(
        [shape_const, t1, r, t2],
        "graph",
        [make_value_info("x", make_tensor_type_proto(TensorProto.FLOAT, [1, 3, 2]))],
        [make_value_info("y", make_tensor_type_proto(TensorProto.FLOAT, [5]))],
        [],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])

    graph_opt = _optimize(model)
    op_types = [n.op_type for n in graph_opt.model.graph.node]
    assert op_types.count("Transpose") == 2


def test_rewrite_when_final_transpose_has_successor():
    shape_const = make_node(
        "Constant",
        [],
        ["shape_val"],
        value=make_tensor("shape_tensor", TensorProto.INT64, [3], [1, 2, 3]),
    )
    t1 = make_node("Transpose", ["x"], ["t1_out"], perm=[0, 2, 1])
    r = make_node("Reshape", ["t1_out", "shape_val"], ["r_out"])
    t2 = make_node("Transpose", ["r_out"], ["t2_out"], perm=[0, 2, 1])
    relu = make_node("Relu", ["t2_out"], ["y"])
    graph = make_graph(
        [shape_const, t1, r, t2, relu],
        "graph",
        [make_value_info("x", make_tensor_type_proto(TensorProto.FLOAT, [1, 3, 2]))],
        [make_value_info("y", make_tensor_type_proto(TensorProto.FLOAT, [1, 3, 2]))],
        [],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])

    graph_opt = _optimize(model)
    op_types = [n.op_type for n in graph_opt.model.graph.node]
    assert "Transpose" not in op_types
    assert "Relu" in op_types
    check_model(graph_opt.model, True)


def test_raise_when_graph_is_not_dag():
    model = _build_basic_chain_model()
    graph = OnnxGraph(model)
    node = next(n for n in graph.model.graph.node if n.op_type == "Transpose")
    rewriter = EliminateDuplicatedTranspose()

    with patch(
        "onnxifier.passes.fusion.eliminate_duplicated_ops.nx.is_directed_acyclic_graph",
        return_value=False,
    ):
        try:
            rewriter.rewrite(graph, [node])
            assert False, "Expected RuntimeError when graph is not DAG"
        except RuntimeError:
            pass


def test_skip_when_shape_is_none_branch():
    model = _build_basic_chain_model()
    graph = OnnxGraph(model)
    node = next(n for n in graph.model.graph.node if n.op_type == "Transpose")
    rewriter = EliminateDuplicatedTranspose()

    with patch.object(OnnxGraph, "tensor_shape", side_effect=[None, [1, 3, 2]]):
        rewriter.rewrite(graph, [node])


def test_skip_when_output_shape_not_all_positive_ints_branch():
    model = _build_basic_chain_model()
    graph = OnnxGraph(model)
    node = next(n for n in graph.model.graph.node if n.op_type == "Transpose")
    rewriter = EliminateDuplicatedTranspose()

    with patch.object(
        OnnxGraph,
        "tensor_shape",
        side_effect=[[1, 3, 2], [1, "N", 2]],
    ):
        rewriter.rewrite(graph, [node])
