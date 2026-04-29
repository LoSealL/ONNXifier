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
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, OnnxGraph, PassManager


def _make_onnx_dq_graph(
    x_data: np.ndarray,
    scale_data: np.ndarray,
    zp_data: np.ndarray | None = None,
    axis: int = 1,
) -> OnnxGraph:
    """Create an ONNX graph with a single DequantizeLinear node."""
    initializers = [from_array(x_data, "x"), from_array(scale_data, "scale")]
    inputs = ["x", "scale"]
    if zp_data is not None:
        initializers.append(from_array(zp_data, "zp"))
        inputs.append("zp")

    dq = make_node(
        "DequantizeLinear",
        inputs,
        ["y"],
        name="dq",
        axis=axis,
    )
    graph = make_graph(
        [dq],
        "test_graph",
        [],
        [make_tensor_value_info("y", TensorProto.FLOAT, list(x_data.shape))],
        initializer=initializers,
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[make_operatorsetid("", 21)],
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def _make_trt_dq_graph(
    x_data: np.ndarray,
    scale_data: np.ndarray,
    zp_data: np.ndarray | None = None,
    axis: int = 1,
) -> OnnxGraph:
    """Create an ONNX graph with a single TRT DequantizeLinear node."""
    initializers = [from_array(x_data, "x"), from_array(scale_data, "scale")]
    inputs = ["x", "scale"]
    if zp_data is not None:
        initializers.append(from_array(zp_data, "zp"))
        inputs.append("zp")

    dq = make_node(
        "DequantizeLinear",
        inputs,
        ["y"],
        name="dq",
        axis=axis,
        domain="trt",
    )
    graph = make_graph(
        [dq],
        "test_graph",
        [],
        [make_tensor_value_info("y", TensorProto.FLOAT, list(x_data.shape))],
        initializer=initializers,
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[make_operatorsetid("", 21), make_operatorsetid("trt", 1)],
    )
    return OnnxGraph(model)


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


class TestTRTDequantizeLinearReplace:
    def test_replace_int8_constant(self):
        x = np.array([0, 64, 127, -128], dtype=np.int8)
        scale = np.array(0.5, dtype=np.float32)
        zp = np.array(0, dtype=np.int8)
        graph = _make_onnx_dq_graph(x, scale, zp)

        graph = PassManager(["trt_dequantize_linear_replace"]).optimize(
            graph, strict=True
        )

        # Original ONNX node should be replaced
        assert _find_node(graph, "DequantizeLinear", "") is None

        # TRT node should exist
        trt_node = _find_node(graph, "DequantizeLinear", "trt")
        assert trt_node is not None
        assert trt_node.input[0] != "x"

        # Constant node with pseudo-quantized float should exist
        const_node = _find_const_node(graph, "pseudo_quant")
        assert const_node is not None

        # Verify the computed float value
        from onnx.numpy_helper import to_array

        float_val = to_array(const_node.attribute[0].t)
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
        # x is a graph input (not initializer) -> should skip
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
            opset_imports=[make_operatorsetid("", 21)],
        )
        onnx.checker.check_model(model)
        graph_obj = OnnxGraph(model)

        graph_obj = PassManager(["trt_dequantize_linear_replace"]).optimize(
            graph_obj, strict=True
        )

        # Original node should remain unchanged
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

        # TRT node should be replaced
        assert _find_node(graph, "DequantizeLinear", "trt") is None

        # ONNX node should exist
        onnx_node = _find_node(graph, "DequantizeLinear", "")
        assert onnx_node is not None
        assert onnx_node.input[0] != "x"

        # Constant node with quantized int8 should exist
        const_node = _find_const_node(graph, "quant_val")
        assert const_node is not None

        from onnx.numpy_helper import to_array

        int_val = to_array(const_node.attribute[0].t)
        expected = np.rint(x / scale + zp).clip(-128, 127).astype(np.int8)
        np.testing.assert_array_equal(int_val, expected)

    def test_restore_non_constant_float(self):
        # x is a graph input (not initializer)
        x_info = make_tensor_value_info("x", TensorProto.FLOAT, [4])
        scale = from_array(np.array(0.5, dtype=np.float32), "scale")
        dq = make_node(
            "DequantizeLinear",
            ["x", "scale"],
            ["y"],
            name="dq",
            axis=1,
            domain="trt",
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
            opset_imports=[
                make_operatorsetid("", 21),
                make_operatorsetid("trt", 1),
            ],
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

        # ONNX -> TRT
        graph = PassManager(["trt_dequantize_linear_replace"]).optimize(
            graph, strict=True
        )
        assert _find_node(graph, "DequantizeLinear", "trt") is not None

        # TRT -> ONNX
        graph = PassManager(["trt_dequantize_linear_to_onnx"]).optimize(
            graph, strict=True
        )
        onnx_node = _find_node(graph, "DequantizeLinear", "")
        assert onnx_node is not None
        assert _find_node(graph, "DequantizeLinear", "trt") is None
