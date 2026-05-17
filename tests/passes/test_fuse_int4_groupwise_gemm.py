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

# pylint: disable=missing-function-docstring,redefined-outer-name

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


def _make_pseudo_quantized_weight(k: int, n: int, group_size: int, *, axis: int = 0):
    """Create pseudo-quantized float weight [k, n] and groupwise scales.

    axis=0: scales [k // group_size, n] expanded to [k, n] for pseudo-quant.
    axis=1: scales [k, n // group_size] expanded to [k, n] for pseudo-quant.
    """
    if axis == 0:
        scales = np.random.rand(k // group_size, n).astype(np.float16)
    else:
        scales = np.random.rand(k, n // group_size).astype(np.float16)
    scales_expanded = np.repeat(scales, group_size, axis=axis)
    int4_vals = np.random.randint(-8, 8, [k, n], dtype=np.int8)
    weight_fake = (int4_vals * scales_expanded).astype(np.float32)
    return weight_fake, scales


def _make_int4_gemm_subgraph(
    with_intermediate=False, with_reshape=False, *, axis: int = 0
):
    """Build a DequantizeLinear -> ** -> MatMul subgraph.

    Three patterns are supported:
    - basic (no intermediates): ``DQ[axis] → MatMul``
    - with_transpose: ``DQ[axis] → Transpose → MatMul``
    - with_reshape: ``DQ[axis] → Reshape → Transpose → MatMul``

    The ``axis`` parameter controls which dimension the scales quantize along:
    - axis=0: scales shape [K//group_size, N], expanded along K
    - axis=1: scales shape [K, N//group_size], expanded along N

    Note: The implementation only fuses two valid patterns:
    - axis=0, non-transposed (basic)
    - axis=1, transposed (with_intermediate or with_reshape)
    """
    group_size = 128
    k, n = 2048, 4096

    qweight, scales = _make_pseudo_quantized_weight(k, n, group_size, axis=axis)
    qweight_init = from_array(qweight, "qweight")
    scales_init = from_array(scales, "scales")

    if with_reshape:
        activation_shape = [1, k]
        output_shape = [1, n]
    elif with_intermediate:
        activation_shape = [1, n]
        output_shape = [1, k]
    else:
        activation_shape = [1, k]
        output_shape = [1, n]

    input_info = make_tensor_value_info(
        "activation", TensorProto.FLOAT, activation_shape
    )
    output_info = make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

    dq = make_node(
        "DequantizeLinear",
        ["qweight", "scales"],
        ["dq_out"],
        name="dq",
        domain="trt",
        axis=axis,
        block_size=group_size,
    )

    nodes = []
    initializers = [qweight_init, scales_init]

    if with_reshape:
        shape_init = from_array(np.array([n, k], dtype=np.int64), "reshape_shape")
        initializers.append(shape_init)
        reshape = make_node(
            "Reshape", ["dq_out", "reshape_shape"], ["dq_r"], name="reshape"
        )
        transp = make_node("Transpose", ["dq_r"], ["dq_t"], name="transp", perm=[1, 0])
        matmul = make_node("MatMul", ["activation", "dq_t"], ["output"], name="matmul")
        nodes = [dq, reshape, transp, matmul]
    elif with_intermediate:
        transp = make_node(
            "Transpose", ["dq_out"], ["dq_t"], name="transp", perm=[1, 0]
        )
        matmul = make_node("MatMul", ["activation", "dq_t"], ["output"], name="matmul")
        nodes = [dq, transp, matmul]
    else:
        matmul = make_node(
            "MatMul", ["activation", "dq_out"], ["output"], name="matmul"
        )
        nodes = [dq, matmul]

    graph = make_graph(
        nodes,
        "int4_gemm_subgraph",
        [input_info],
        [output_info],
        initializer=initializers,
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[
            make_operatorsetid("", 21),
            make_operatorsetid("trt", 1),
        ],
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def _find_plugin(graph):
    for node_name in graph:
        node = graph.nodes[node_name]["pb"]
        if node.op_type == "Int4GroupwiseGemmPlugin":
            return node
    return None


def _get_constant_by_output(graph, output_name):
    for node_name in graph:
        node = graph.nodes[node_name]["pb"]
        if node.op_type == "Constant" and output_name in node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    return attr.t
    return None


def _check_shapes(
    graph, plugin, exp_qweight, exp_scales, gemm_k, gemm_n, group_size=128
):
    """Verify plugin attributes and packed/scales constant tensor shapes.

    ``exp_qweight`` is the expected dims of the packed int4 constant.
    ``exp_scales`` is the expected dims of the scales constant.
    Packed shape depends on _pack_int4: (N//2, K) in int8.
    Scales shape is always [gemm_k // group_size, gemm_n].
    """
    attrs = {a.name: a.i for a in plugin.attribute}
    assert attrs["gemm_k"] == gemm_k
    assert attrs["gemm_n"] == gemm_n
    assert attrs["group_size"] == group_size

    assert plugin.input[0] == "activation"

    packed = _get_constant_by_output(graph, plugin.input[1])
    assert packed is not None
    assert packed.data_type == TensorProto.INT8
    assert list(packed.dims) == exp_qweight

    scales = _get_constant_by_output(graph, plugin.input[2])
    assert scales is not None
    assert list(scales.dims) == exp_scales


def test_fuse_int4_groupwise_gemm_basic():
    # DQ[axis=0] → MatMul: weight [2048, 4096], scales [16, 4096]
    # Packed: _pack_int4([4096, 2048]) → [2048, 2048] (N//2, K)
    graph = _make_int4_gemm_subgraph(with_intermediate=False)
    graph = PassManager(["fuse_int4_groupwise_gemm"]).optimize(graph, strict=True)

    plugin = _find_plugin(graph)
    assert plugin is not None
    assert plugin.domain == "trt"
    assert list(plugin.output) == ["output"]
    assert not any(graph.nodes[n]["pb"].op_type == "DequantizeLinear" for n in graph)
    assert not any(graph.nodes[n]["pb"].op_type == "MatMul" for n in graph)

    _check_shapes(graph, plugin, [2048, 2048], [16, 4096], 2048, 4096)


def test_fuse_int4_groupwise_gemm_with_transpose():
    # DQ[axis=1] → Transpose → MatMul: weight [2048, 4096] transposed to [4096, 2048]
    # gemm_k=4096, gemm_n=2048
    # Packed: _pack_int4([2048, 4096]) → [1024, 4096] (N//2, K)
    # Scales: [32, 2048] = [gemm_k//128, gemm_n]
    graph = _make_int4_gemm_subgraph(with_intermediate=True, axis=1)
    graph = PassManager(["fuse_int4_groupwise_gemm"]).optimize(graph, strict=True)

    plugin = _find_plugin(graph)
    assert plugin is not None
    assert not any(graph.nodes[n]["pb"].op_type == "Transpose" for n in graph)

    _check_shapes(graph, plugin, [1024, 4096], [32, 2048], 4096, 2048)


def test_fuse_int4_groupwise_gemm_with_reshape():
    # DQ[axis=1] → Reshape → Transpose → MatMul
    # Reshape+Transpose effectively undo each other: weight ends as [2048, 4096]
    # gemm_k=2048, gemm_n=4096
    # Packed: _pack_int4([4096, 2048]) → [2048, 2048] (N//2, K)
    # Scales: [16, 4096] = [gemm_k//128, gemm_n]
    graph = _make_int4_gemm_subgraph(with_reshape=True, axis=1)
    graph = PassManager(["fuse_int4_groupwise_gemm"]).optimize(graph, strict=True)

    plugin = _find_plugin(graph)
    assert plugin is not None
    assert not any(graph.nodes[n]["pb"].op_type == "Reshape" for n in graph)
    assert not any(graph.nodes[n]["pb"].op_type == "Transpose" for n in graph)

    _check_shapes(graph, plugin, [2048, 2048], [16, 4096], 2048, 4096)


def test_fuse_int4_groupwise_gemm_idempotent():
    graph = _make_int4_gemm_subgraph(with_intermediate=False)
    graph = PassManager(["fuse_int4_groupwise_gemm"]).optimize(graph, strict=True)
    count_after_first = len(graph)
    graph = PassManager(["fuse_int4_groupwise_gemm"]).optimize(graph, strict=True)
    assert len(graph) == count_after_first
