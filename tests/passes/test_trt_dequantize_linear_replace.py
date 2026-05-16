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

import numpy as np
import onnx
import pytest
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array, to_array

from onnxifier import ONNXIFIER_IR_VERSION, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator

# ============================================================
# numpy reference
# ============================================================


def _dequantize_linear_numpy(
    x: np.ndarray,
    scale: np.ndarray,
    zero_point: np.ndarray | None = None,
    axis: int = 1,
    block_size: int = 0,
) -> np.ndarray:
    x_float = x.astype(np.float32)
    if block_size > 0 and block_size < x.shape[axis]:
        scale = np.repeat(scale, block_size, axis=axis)
        scale = scale[tuple(slice(0, s) for s in x.shape)]
        if zero_point is not None:
            zero_point = np.repeat(zero_point, block_size, axis=axis)
            zero_point = zero_point[tuple(slice(0, s) for s in x.shape)]
    elif scale.ndim == 1 and scale.size > 1:
        shape = [1] * x.ndim
        shape[axis] = scale.shape[0]
        scale = scale.reshape(shape)
        if zero_point is not None and zero_point.ndim == 1:
            zero_point = zero_point.reshape(shape)
    if zero_point is not None:
        x_float = x_float - zero_point.astype(np.float32)
    return x_float * scale.astype(np.float32)


# ============================================================
# graph builders
# ============================================================


def _make_onnx_dq_graph(
    x_data: np.ndarray,
    scale_data: np.ndarray,
    zp_data: np.ndarray | None = None,
    axis: int | None = None,
    block_size: int | None = None,
) -> OnnxGraph:
    initializers = [from_array(x_data, "x"), from_array(scale_data, "scale")]
    inputs = ["x", "scale"]
    if zp_data is not None:
        initializers.append(from_array(zp_data, "zp"))
        inputs.append("zp")

    kw = {}
    if axis is not None:
        kw["axis"] = axis
    if block_size is not None:
        kw["block_size"] = block_size

    dq = make_node("DequantizeLinear", inputs, ["y"], name="dq", **kw)
    out_dtype = TensorProto.FLOAT16 if x_data.dtype == np.float16 else TensorProto.FLOAT
    graph = make_graph(
        [dq],
        "g",
        [],
        [make_tensor_value_info("y", out_dtype, list(x_data.shape))],
        initializer=initializers,
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[make_operatorsetid("", 23)],
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def _make_trt_dq_graph(
    x_data: np.ndarray,
    scale_data: np.ndarray,
    zp_data: np.ndarray | None = None,
    axis: int | None = None,
    block_size: int | None = None,
) -> OnnxGraph:
    initializers = [from_array(x_data, "x"), from_array(scale_data, "scale")]
    inputs = ["x", "scale"]
    if zp_data is not None:
        initializers.append(from_array(zp_data, "zp"))
        inputs.append("zp")

    kw = {}
    if axis is not None:
        kw["axis"] = axis
    if block_size is not None:
        kw["block_size"] = block_size

    dq = make_node("DequantizeLinear", inputs, ["y"], name="dq", domain="trt", **kw)
    graph = make_graph(
        [dq],
        "g",
        [],
        [make_tensor_value_info("y", TensorProto.FLOAT, list(x_data.shape))],
        initializer=initializers,
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[make_operatorsetid("", 23), make_operatorsetid("trt", 1)],
    )
    return OnnxGraph(model)


# ============================================================
# helpers
# ============================================================


def _find_node(graph: OnnxGraph, op_type: str, domain: str | None = None):
    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type == op_type and (domain is None or node.domain == domain):
            return node
    return None


def _find_const_node(graph: OnnxGraph, name_hint: str):
    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type == "Constant" and name_hint in node.name:
            return node
    return None


def _get_attr(node: onnx.NodeProto, name: str, default=None):
    for a in node.attribute:
        if a.name == name:
            return a.i
    return default


def _eval_output(graph: OnnxGraph) -> np.ndarray:
    runner = Evaluator(graph.model, "onnx")
    feeds = {
        inp.name: np.zeros(
            [d.dim_value for d in inp.type.tensor_type.shape.dim], dtype=np.float32
        )
        for inp in graph.model.graph.input
    }
    return runner(["y"], feeds)[0]


# ============================================================
# 1.1 — axis / block_size diversity (parametrized)
# ============================================================


AXIS_BLOCK_CASES = [
    pytest.param(
        np.array([[0, 64, 127], [-128, -64, 0]], dtype=np.int8),
        np.array([0.5, 1.0], dtype=np.float32),
        np.array([1, -1], dtype=np.int8),
        0,
        0,
        id="axis_0_no_block",
    ),
    pytest.param(
        np.array([[0, 64, 32], [-128, -64, 0]], dtype=np.int8),
        np.array([0.5, 1.0, 1.5], dtype=np.float32),
        None,
        1,
        0,
        id="axis_1_default",
    ),
]


@pytest.mark.parametrize("x,scale,zp,axis,bs", AXIS_BLOCK_CASES)
def test_axis_block_diversity(x, scale, zp, axis, bs):
    graph = _make_onnx_dq_graph(x, scale, zp, axis=axis, block_size=bs)
    expected = _eval_output(graph)
    graph = PassManager(["trt_dequantize_linear_replace"]).optimize(graph, strict=True)

    trt_node = _find_node(graph, "DequantizeLinear", "trt")
    assert trt_node is not None
    assert _get_attr(trt_node, "axis") == axis

    const_node = _find_const_node(graph, "pseudo_quant")
    float_val = to_array(const_node.attribute[0].t)
    np.testing.assert_allclose(float_val, expected)


@pytest.mark.parametrize(
    "shape,axis,bs",
    [
        pytest.param((128, 64), 0, 64, id="axis_0_bs_64"),
        pytest.param((64, 128), 1, 32, id="axis_1_bs_32"),
    ],
)
def test_axis_block_size_groupwise(shape, axis, bs):
    rng = np.random.RandomState(42)
    k, n = shape
    x = rng.randint(-128, 128, shape, dtype=np.int8)
    scale = rng.randn(k // bs if axis == 0 else k, n // bs if axis == 1 else n)
    scale = scale.astype(np.float32) * 0.1
    zp = np.zeros_like(scale, dtype=np.int8)

    graph = _make_onnx_dq_graph(x, scale, zp, axis=axis, block_size=bs)
    expected = _eval_output(graph)
    graph = PassManager(["trt_dequantize_linear_replace"]).optimize(graph, strict=True)

    trt_node = _find_node(graph, "DequantizeLinear", "trt")
    assert trt_node is not None
    assert _get_attr(trt_node, "axis") == axis
    assert _get_attr(trt_node, "block_size") == bs

    float_val = to_array(_find_const_node(graph, "pseudo_quant").attribute[0].t)
    np.testing.assert_allclose(float_val, expected)


# ============================================================
# 1.2 — TRT DQ axis inference (parametrized)
# ============================================================


AXIS_INFERENCE_CASES = [
    pytest.param((128, 64), (128, 1), None, 0, 0, id="infer_axis_0_from_2d_scale"),
    pytest.param((64, 32), (1, 32), None, 1, 0, id="infer_axis_1_from_2d_scale"),
    pytest.param((256, 128), (256, 1), 1, 0, 0, id="wrong_axis_corrected"),
    pytest.param((8, 12, 16), (8, 12, 1), None, 2, 16, id="infer_axis_2_3d"),
]


@pytest.mark.parametrize(
    "x_shape,scale_shape,wrong_axis,exp_axis,block_size", AXIS_INFERENCE_CASES
)
def test_trt_dq_axis_inference(x_shape, scale_shape, wrong_axis, exp_axis, block_size):
    x = np.random.randint(-128, 128, x_shape, dtype=np.int8).astype(np.float32)
    scale = np.random.randn(*scale_shape).astype(np.float32) * 0.1
    graph = _make_trt_dq_graph(x, scale, axis=wrong_axis, block_size=block_size or None)

    graph = PassManager(["trt_dequantize_linear_to_onnx"]).optimize(graph, strict=True)
    onnx_node = _find_node(graph, "DequantizeLinear", "")
    assert onnx_node is not None
    assert _get_attr(onnx_node, "axis") == exp_axis


# ============================================================
# 1.3 — numpy reference accuracy (parametrized)
# ============================================================


_REF_RNG = np.random.RandomState(123)

ACCURACY_CASES = [
    pytest.param(
        _REF_RNG.randint(-128, 128, [32], dtype=np.int8),
        np.array(0.25, dtype=np.float32),
        np.array(0, dtype=np.int8),
        None,
        0,
        id="per_tensor",
    ),
    pytest.param(
        _REF_RNG.randint(-128, 128, [16, 8], dtype=np.int8),
        (_REF_RNG.randn(16).astype(np.float32) * 0.5),
        np.zeros([16], dtype=np.int8),
        0,
        0,
        id="axis_0",
    ),
    pytest.param(
        _REF_RNG.randint(-128, 128, [8, 16], dtype=np.int8),
        (_REF_RNG.randn(16).astype(np.float32) * 0.5),
        np.zeros([16], dtype=np.int8),
        1,
        0,
        id="axis_1",
    ),
    pytest.param(
        _REF_RNG.randint(-128, 128, [4, 8, 12], dtype=np.int8),
        (_REF_RNG.randn(12).astype(np.float32) * 0.5),
        np.zeros([12], dtype=np.int8),
        2,
        0,
        id="axis_2",
    ),
    pytest.param(
        _REF_RNG.randint(-128, 128, [2, 4, 6, 8], dtype=np.int8),
        (_REF_RNG.randn(8).astype(np.float32) * 0.5),
        np.zeros([8], dtype=np.int8),
        3,
        0,
        id="axis_3",
    ),
    pytest.param(
        _REF_RNG.randint(-128, 128, [1, 2, 3, 4, 5], dtype=np.int8),
        (_REF_RNG.randn(5).astype(np.float32) * 0.5),
        np.zeros([5], dtype=np.int8),
        4,
        0,
        id="axis_4",
    ),
    pytest.param(
        _REF_RNG.randint(-128, 128, [256, 64], dtype=np.int8),
        (_REF_RNG.randn(4, 64).astype(np.float32) * 0.1),
        np.zeros([4, 64], dtype=np.int8),
        0,
        64,
        id="block_axis_0",
    ),
    pytest.param(
        _REF_RNG.randint(-128, 128, [32, 128], dtype=np.int8),
        (_REF_RNG.randn(32, 2).astype(np.float32) * 0.1),
        np.zeros([32, 2], dtype=np.int8),
        1,
        64,
        id="block_axis_1",
    ),
    pytest.param(
        _REF_RNG.randint(-128, 128, [8, 12, 64], dtype=np.int8),
        (_REF_RNG.randn(8, 12, 1).astype(np.float32) * 0.1),
        np.zeros([8, 12, 1], dtype=np.int8),
        2,
        64,
        id="block_axis_2",
    ),
]


@pytest.mark.parametrize("x,scale,zp,axis,bs", ACCURACY_CASES)
def test_accuracy_numpy_reference(x, scale, zp, axis, bs):
    graph = _make_onnx_dq_graph(x, scale, zp, axis=axis, block_size=bs or None)
    actual = _eval_output(graph)
    expected = _dequantize_linear_numpy(
        x, scale, zp, axis=0 if axis is None else axis, block_size=bs
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


# ============================================================
# legacy class-based tests (unique logic per test)
# ============================================================


class TestTRTDequantizeLinearReplace:
    def test_replace_int8_constant(self):
        x = np.array([0, 64, 127, -128], dtype=np.int8)
        scale = np.array(0.5, dtype=np.float32)
        zp = np.array(0, dtype=np.int8)
        graph = _make_onnx_dq_graph(x, scale, zp)

        graph = PassManager(["trt_dequantize_linear_replace"]).optimize(
            graph, strict=True
        )
        assert _find_node(graph, "DequantizeLinear", "") is None
        trt_node = _find_node(graph, "DequantizeLinear", "trt")
        assert trt_node is not None
        assert trt_node.input[0] != "x"

        float_val = to_array(_find_const_node(graph, "pseudo_quant").attribute[0].t)
        expected = (x.astype(np.float32) - zp.astype(np.float32)) * scale
        np.testing.assert_allclose(float_val, expected)

    def test_replace_float_input_unchanged(self):
        x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        scale = np.array(0.5, dtype=np.float32)
        graph = _make_onnx_dq_graph(x, scale)

        graph = PassManager(["trt_dequantize_linear_replace"]).optimize(
            graph, strict=True
        )
        trt_node = _find_node(graph, "DequantizeLinear", "trt")
        assert trt_node is not None
        assert trt_node.input[0] == "x"

    def test_skip_non_constant_input(self):
        x_info = make_tensor_value_info("x", TensorProto.INT8, [4])
        scale = from_array(np.array(0.5, dtype=np.float32), "scale")
        dq = make_node("DequantizeLinear", ["x", "scale"], ["y"], name="dq", axis=1)
        graph = make_graph(
            [dq],
            "test_graph",
            [x_info],
            [make_tensor_value_info("y", TensorProto.FLOAT, [4])],
            initializer=[scale],
        )
        model = make_model(
            graph,
            ir_version=ONNXIFIER_IR_VERSION,
            opset_imports=[make_operatorsetid("", 23)],
        )
        onnx.checker.check_model(model)
        graph_obj = OnnxGraph(model)

        graph_obj = PassManager(["trt_dequantize_linear_replace"]).optimize(
            graph_obj, strict=True
        )
        assert _find_node(graph_obj, "DequantizeLinear", "") is not None
        assert _find_node(graph_obj, "DequantizeLinear", "trt") is None


class TestTRTDequantizeLinearToOnnx:
    def test_restore_float_constant(self):
        x = np.array([0.0, 32.0, 63.5, -64.0], dtype=np.float32)
        scale = np.array(0.5, dtype=np.float32)
        zp = np.array(0, dtype=np.float32)
        graph = _make_trt_dq_graph(x, scale, zp)

        graph = PassManager(["trt_dequantize_linear_to_onnx"]).optimize(
            graph, strict=True
        )
        assert _find_node(graph, "DequantizeLinear", "trt") is None
        onnx_node = _find_node(graph, "DequantizeLinear", "")
        assert onnx_node is not None
        assert onnx_node.input[0] != "x"

        scale = to_array(_find_const_node(graph, "scale").attribute[0].t)
        assert scale.squeeze().ndim <= 1
        int_val = to_array(_find_const_node(graph, "weight").attribute[0].t)
        expected = np.rint(x / scale + zp).clip(-128, 127).astype(np.int8)
        np.testing.assert_array_equal(int_val, expected)

    def test_restore_non_constant_float(self):
        x_info = make_tensor_value_info("x", TensorProto.FLOAT, [4])
        scale = from_array(np.array(0.5, dtype=np.float32), "scale")
        dq = make_node(
            "DequantizeLinear", ["x", "scale"], ["y"], name="dq", axis=1, domain="trt"
        )
        graph = make_graph(
            [dq],
            "test_graph",
            [x_info],
            [make_tensor_value_info("y", TensorProto.FLOAT, [4])],
            initializer=[scale],
        )
        model = make_model(
            graph,
            ir_version=ONNXIFIER_IR_VERSION,
            opset_imports=[make_operatorsetid("", 23), make_operatorsetid("trt", 1)],
        )
        graph_obj = OnnxGraph(model)

        graph_obj = PassManager(["trt_dequantize_linear_to_onnx"]).optimize(
            graph_obj, strict=True
        )
        onnx_node = _find_node(graph_obj, "DequantizeLinear", "")
        assert onnx_node is not None
        assert onnx_node.input[0] == "x"


class TestRoundTrip:
    def test_round_trip(self):
        x_int = np.array([0, 64, 127, -128], dtype=np.int8)
        scale = np.array(0.5, dtype=np.float32)
        zp = np.array(0, dtype=np.int8)
        graph = _make_onnx_dq_graph(x_int, scale, zp)

        graph = PassManager(["trt_dequantize_linear_replace"]).optimize(
            graph, strict=True
        )
        assert _find_node(graph, "DequantizeLinear", "trt") is not None

        graph = PassManager(["trt_dequantize_linear_to_onnx"]).optimize(
            graph, strict=True
        )
        onnx_node = _find_node(graph, "DequantizeLinear", "")
        assert onnx_node is not None
        assert _find_node(graph, "DequantizeLinear", "trt") is None

        recovered = to_array(_find_const_node(graph, "weight").attribute[0].t)
        np.testing.assert_array_equal(recovered, x_int)
