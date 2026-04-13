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
from onnx import TensorProto, numpy_helper
from onnx.checker import check_model
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_value_info,
)

from onnxifier import ONNXIFIER_OPSET, OnnxGraph, PassManager


def _optimize(model: onnx.ModelProto, pass_name: str) -> OnnxGraph:
    graph = OnnxGraph(model)
    pm = PassManager([pass_name])
    return pm.optimize(graph, strict=True)


def _optimize_slice(model: onnx.ModelProto) -> OnnxGraph:
    return _optimize(model, "eliminate_nop_slice")


def _optimize_concat(model: onnx.ModelProto) -> OnnxGraph:
    return _optimize(model, "eliminate_nop_concat")


def _optimize_transpose(model: onnx.ModelProto) -> OnnxGraph:
    return _optimize(model, "eliminate_nop_transpose")


def _optimize_pad(model: onnx.ModelProto) -> OnnxGraph:
    return _optimize(model, "eliminate_nop_pad")


# ─────────────────────────────── helpers ───────────────────────────────────


def _make_init(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


# ══════════════════════════════════════════════════════════════════════════
# Slice tests
# ══════════════════════════════════════════════════════════════════════════


def _build_slice_graph(
    in_shape: list[int],
    out_shape: list[int],
    starts: list[int],
    ends: list[int],
    axes: list[int] | None = None,
    steps: list[int] | None = None,
) -> onnx.ModelProto:
    """input → Relu → Slice(…) → Relu → output."""
    slice_inputs = ["relu0_out", "slice_starts", "slice_ends"]
    initializers = [
        _make_init("slice_starts", np.array(starts, np.int64)),
        _make_init("slice_ends", np.array(ends, np.int64)),
    ]
    if axes is not None:
        initializers.append(_make_init("slice_axes", np.array(axes, np.int64)))
        slice_inputs.append("slice_axes")
    if steps is not None:
        if axes is None:
            slice_inputs.append("")  # axes placeholder
        initializers.append(_make_init("slice_steps", np.array(steps, np.int64)))
        slice_inputs.append("slice_steps")

    nodes = [
        make_node("Relu", ["x"], ["relu0_out"], "relu0"),
        make_node("Slice", slice_inputs, ["slice_out"], "slice"),
        make_node("Relu", ["slice_out"], ["y"], "relu1"),
    ]
    graph = make_graph(
        nodes,
        "graph",
        [make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
        [make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
        initializers,
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def test_slice_oob_end_eliminated():
    """a[:100] where shape=[16] → clamps to full extent → eliminate."""
    model = _build_slice_graph([16], [16], starts=[0], ends=[100])
    graph = _optimize_slice(model)
    assert "slice" not in graph
    assert len(graph.nodes) == 2  # only two Relu nodes remain
    check_model(graph.model)


def test_slice_exact_end_eliminated():
    """a[0:16] on shape=[16] → exact full coverage → eliminate."""
    model = _build_slice_graph([16], [16], starts=[0], ends=[16])
    graph = _optimize_slice(model)
    assert "slice" not in graph
    check_model(graph.model)


def test_slice_with_explicit_unit_step_eliminated():
    """a[0:16:1] on shape=[16] → same with explicit step=1 → eliminate."""
    model = _build_slice_graph([16], [16], starts=[0], ends=[16], steps=[1])
    graph = _optimize_slice(model)
    assert "slice" not in graph
    check_model(graph.model)


def test_slice_2d_full_both_axes_eliminated():
    """a[0:4, 0:8] on shape=[4, 8] → full on both axes → eliminate."""
    model = _build_slice_graph(
        [4, 8],
        [4, 8],
        starts=[0, 0],
        ends=[4, 8],
        axes=[0, 1],
    )
    graph = _optimize_slice(model)
    assert "slice" not in graph
    check_model(graph.model)


def test_slice_partial_end_not_eliminated():
    """a[0:5] on shape=[16] → partial → keep."""
    model = _build_slice_graph([16], [5], starts=[0], ends=[5])
    graph = _optimize_slice(model)
    assert "slice" in graph


def test_slice_nonzero_start_not_eliminated():
    """a[2:16] on shape=[16] → start!=0 after clamping → keep."""
    model = _build_slice_graph([16], [14], starts=[2], ends=[16])
    graph = _optimize_slice(model)
    assert "slice" in graph


def test_slice_step_two_not_eliminated():
    """a[0:16:2] → step!=1 → keep."""
    model = _build_slice_graph([16], [8], starts=[0], ends=[16], steps=[2])
    graph = _optimize_slice(model)
    assert "slice" in graph


def test_slice_graph_output_eliminated():
    """Slice is a graph output: predecessor's output should be renamed."""
    init = _make_init("slice_starts", np.array([0], np.int64))
    init2 = _make_init("slice_ends", np.array([999], np.int64))
    relu = make_node("Relu", ["x"], ["relu_out"], "relu")
    slice_nd = make_node(
        "Slice", ["relu_out", "slice_starts", "slice_ends"], ["y"], "slice"
    )
    graph = make_graph(
        [relu, slice_nd],
        "graph",
        [make_tensor_value_info("x", TensorProto.FLOAT, [16])],
        [make_tensor_value_info("y", TensorProto.FLOAT, [16])],
        [init, init2],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])
    g = _optimize_slice(model)
    assert "slice" not in g
    assert "y" in g.outputs
    check_model(g.model)


def _build_split_graph(
    in_shape: list[int],
    out_shape: list[int],
    split: list[int] | None,
    outputs: list[str],
    axis: int = 0,
) -> onnx.ModelProto:
    """input → Relu → Split(…) -> Relu(s) -> output(s)."""
    split_inputs = ["relu0_out"]
    initializers = []
    if split is not None:
        split_inputs.append("split_spec")
        initializers.append(_make_init("split_spec", np.array(split, np.int64)))

    relu0 = make_node("Relu", ["x"], ["relu0_out"], "relu0")
    split_node = make_node("Split", split_inputs, outputs, "split", axis=axis)
    post_nodes = []
    graph_outputs = []
    for i, out in enumerate(outputs):
        y = f"y{i}"
        post_nodes.append(make_node("Relu", [out], [y], f"relu{i + 1}"))
        graph_outputs.append(make_tensor_value_info(y, TensorProto.FLOAT, out_shape))

    graph = make_graph(
        [relu0, split_node] + post_nodes,
        "graph",
        [make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
        graph_outputs,
        initializers,
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def test_split_single_output_full_size_eliminated():
    """Split with one output and split=[dim] is a pass-through."""
    model = _build_split_graph([16], [16], split=[16], outputs=["split_out"])
    graph = _optimize_slice(model)
    assert "split" not in graph
    check_model(graph.model)


def test_split_single_output_without_split_eliminated():
    """Split with one output and no split spec is a pass-through."""
    model = _build_split_graph([16], [16], split=None, outputs=["split_out"])
    graph = _optimize_slice(model)
    assert "split" not in graph
    check_model(graph.model)


def test_split_multi_output_not_eliminated():
    """Split with multiple outputs is not a no-op and must be kept."""
    model = _build_split_graph(
        [16],
        [8],
        split=[8, 8],
        outputs=["split_out0", "split_out1"],
    )
    graph = _optimize_slice(model)
    assert "split" in graph


# ══════════════════════════════════════════════════════════════════════════
# Concat tests
# ══════════════════════════════════════════════════════════════════════════


def _build_concat_graph(n_inputs: int, axis: int = 0) -> onnx.ModelProto:
    """Build a graph with N Relu nodes feeding a Concat, then another Relu."""
    relu_nodes = [
        make_node("Relu", [f"x{i}"], [f"relu_out{i}"], f"relu{i}")
        for i in range(n_inputs)
    ]
    concat = make_node(
        "Concat",
        [f"relu_out{i}" for i in range(n_inputs)],
        ["concat_out"],
        "concat",
        axis=axis,
    )
    relu_final = make_node("Relu", ["concat_out"], ["y"], "relu_final")
    in_shape = [4, 8]
    out_shape = [4 * n_inputs, 8] if axis == 0 else [4, 8 * n_inputs]
    inputs = [
        make_tensor_value_info(f"x{i}", TensorProto.FLOAT, in_shape)
        for i in range(n_inputs)
    ]
    graph = make_graph(
        relu_nodes + [concat, relu_final],
        "graph",
        inputs,
        [make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def test_concat_single_input_eliminated():
    """Concat with 1 input → trivially a pass-through → eliminate."""
    model = _build_concat_graph(1)
    graph = _optimize_concat(model)
    assert "concat" not in graph
    assert len(graph.nodes) == 2  # relu0 + relu_final
    check_model(graph.model)


def test_concat_two_inputs_not_eliminated():
    """Concat with 2 inputs → real concat → keep."""
    model = _build_concat_graph(2)
    graph = _optimize_concat(model)
    assert "concat" in graph


def test_concat_three_inputs_not_eliminated():
    """Concat with 3 inputs → real concat → keep."""
    model = _build_concat_graph(3)
    graph = _optimize_concat(model)
    assert "concat" in graph


# ══════════════════════════════════════════════════════════════════════════
# Transpose tests
# ══════════════════════════════════════════════════════════════════════════


def _build_transpose_graph(
    shape: list[int], perm: list[int] | None, out_shape: list[int]
) -> onnx.ModelProto:
    """input → Relu → Transpose(perm) → Relu → output."""
    attrs = {"perm": perm} if perm is not None else {}
    relu0 = make_node("Relu", ["x"], ["relu0_out"], "relu0")
    transpose = make_node(
        "Transpose", ["relu0_out"], ["trans_out"], "transpose", **attrs
    )
    relu1 = make_node("Relu", ["trans_out"], ["y"], "relu1")
    graph = make_graph(
        [relu0, transpose, relu1],
        "graph",
        [make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def test_transpose_identity_perm_4d_eliminated():
    """Transpose(perm=[0,1,2,3]) → identity permutation → eliminate."""
    model = _build_transpose_graph([1, 3, 4, 5], [0, 1, 2, 3], [1, 3, 4, 5])
    graph = _optimize_transpose(model)
    assert "transpose" not in graph
    assert len(graph.nodes) == 2
    check_model(graph.model)


def test_transpose_identity_perm_2d_eliminated():
    """Transpose(perm=[0,1]) → identity → eliminate."""
    model = _build_transpose_graph([7, 11], [0, 1], [7, 11])
    graph = _optimize_transpose(model)
    assert "transpose" not in graph
    check_model(graph.model)


def test_transpose_non_identity_not_eliminated():
    """Transpose(perm=[0,2,1,3]) → real permutation → keep."""
    model = _build_transpose_graph([1, 3, 4, 5], [0, 2, 1, 3], [1, 4, 3, 5])
    graph = _optimize_transpose(model)
    assert "transpose" in graph


def test_transpose_reverse_not_eliminated():
    """Transpose(perm=[3,2,1,0]) → reverse → keep."""
    model = _build_transpose_graph([1, 3, 4, 5], [3, 2, 1, 0], [5, 4, 3, 1])
    graph = _optimize_transpose(model)
    assert "transpose" in graph


def test_transpose_graph_output_eliminated():
    """Transpose that feeds the graph output → rename pred output."""
    relu = make_node("Relu", ["x"], ["relu_out"], "relu")
    transpose = make_node(
        "Transpose", ["relu_out"], ["y"], "transpose", perm=[0, 1, 2, 3]
    )
    graph = make_graph(
        [relu, transpose],
        "graph",
        [make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 5])],
        [make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 5])],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])
    g = _optimize_transpose(model)
    assert "transpose" not in g
    assert "y" in g.outputs
    check_model(g.model)


# ══════════════════════════════════════════════════════════════════════════
# Pad tests
# ══════════════════════════════════════════════════════════════════════════


def _build_pad_graph(
    in_shape: list[int],
    out_shape: list[int],
    pads: list[int],
) -> onnx.ModelProto:
    """input → Relu → Pad(pads) → Relu → output."""
    relu0 = make_node("Relu", ["x"], ["relu0_out"], "relu0")
    pad = make_node("Pad", ["relu0_out", "pad_pads"], ["pad_out"], "pad")
    relu1 = make_node("Relu", ["pad_out"], ["y"], "relu1")
    graph = make_graph(
        [relu0, pad, relu1],
        "graph",
        [make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
        [make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
        [_make_init("pad_pads", np.array(pads, np.int64))],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def test_pad_all_zeros_eliminated():
    """Pad(pads=[0,0,0,0,0,0,0,0]) on [1,3,4,5] → identity → eliminate."""
    model = _build_pad_graph([1, 3, 4, 5], [1, 3, 4, 5], [0, 0, 0, 0, 0, 0, 0, 0])
    graph = _optimize_pad(model)
    assert "pad" not in graph
    assert len(graph.nodes) == 2
    check_model(graph.model)


def test_pad_nonzero_not_eliminated():
    """Pad(pads=[0,0,0,1,0,0,0,1]) → non-zero padding → keep."""
    model = _build_pad_graph([1, 3, 4, 5], [1, 3, 4, 7], [0, 0, 0, 1, 0, 0, 0, 1])
    graph = _optimize_pad(model)
    assert "pad" in graph


def test_pad_some_nonzero_not_eliminated():
    """Pad where only one value is nonzero → keep."""
    model = _build_pad_graph([1, 3, 4, 5], [1, 3, 5, 5], [0, 0, 1, 0, 0, 0, 0, 0])
    graph = _optimize_pad(model)
    assert "pad" in graph
