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


def _make_pseudo_quantized_weight(k: int, n: int, group_size: int):
    """Create pseudo-quantized float weight and float16 scales."""
    scales = np.random.rand(k // group_size, n).astype(np.float16)
    scales_expanded = np.repeat(scales, group_size, axis=0)
    int4_vals = np.random.randint(-8, 8, [k, n], dtype=np.int8)
    weight_fake = (int4_vals * scales_expanded).astype(np.float32)
    return weight_fake, scales


def _make_int4_gemm_subgraph(with_intermediate=False, with_reshape=False):
    """Build a DequantizeLinear -> ** -> MatMul subgraph."""
    group_size = 128
    k, n = 2048, 4096

    qweight, scales = _make_pseudo_quantized_weight(k, n, group_size)
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
        axis=0,
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


def _get_packed_constant(graph, plugin):
    packed_name = plugin.input[1]
    for node_name in graph:
        node = graph.nodes[node_name]["pb"]
        if node.op_type == "Constant" and packed_name in node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    return attr.t
    return None


def test_fuse_int4_groupwise_gemm_basic():
    graph = _make_int4_gemm_subgraph(with_intermediate=False)

    graph = PassManager(["fuse_int4_groupwise_gemm"]).optimize(graph, strict=True)

    plugin = _find_plugin(graph)
    assert plugin is not None
    assert plugin.domain == "trt"
    assert list(plugin.output) == ["output"]
    attrs = {a.name: a.i for a in plugin.attribute}
    assert attrs["gemm_k"] == 2048
    assert attrs["gemm_n"] == 4096
    assert attrs["group_size"] == 128

    assert not any(graph.nodes[n]["pb"].op_type == "DequantizeLinear" for n in graph)
    assert not any(graph.nodes[n]["pb"].op_type == "MatMul" for n in graph)

    assert plugin.input[2] == "scales"
    assert plugin.input[0] == "activation"

    packed_tensor = _get_packed_constant(graph, plugin)
    assert packed_tensor is not None
    assert packed_tensor.data_type == TensorProto.INT8
    assert list(packed_tensor.dims) == [1024, 4096]


def test_fuse_int4_groupwise_gemm_with_transpose():
    graph = _make_int4_gemm_subgraph(with_intermediate=True)

    graph = PassManager(["fuse_int4_groupwise_gemm"]).optimize(graph, strict=True)

    plugin = _find_plugin(graph)
    assert plugin is not None
    attrs = {a.name: a.i for a in plugin.attribute}
    assert attrs["gemm_k"] == 4096
    assert attrs["gemm_n"] == 2048
    assert attrs["group_size"] == 128

    assert not any(graph.nodes[n]["pb"].op_type == "Transpose" for n in graph)

    packed_tensor = _get_packed_constant(graph, plugin)
    assert packed_tensor is not None
    assert packed_tensor.data_type == TensorProto.INT8
    assert list(packed_tensor.dims) == [1024, 4096]


def test_fuse_int4_groupwise_gemm_with_reshape():
    graph = _make_int4_gemm_subgraph(with_reshape=True)

    graph = PassManager(["fuse_int4_groupwise_gemm"]).optimize(graph, strict=True)

    plugin = _find_plugin(graph)
    assert plugin is not None
    attrs = {a.name: a.i for a in plugin.attribute}
    assert attrs["gemm_k"] == 2048
    assert attrs["gemm_n"] == 4096
    assert attrs["group_size"] == 128

    assert not any(graph.nodes[n]["pb"].op_type == "Reshape" for n in graph)
    assert not any(graph.nodes[n]["pb"].op_type == "Transpose" for n in graph)

    packed_tensor = _get_packed_constant(graph, plugin)
    assert packed_tensor is not None
    assert packed_tensor.data_type == TensorProto.INT8
    assert list(packed_tensor.dims) == [1024, 4096]


def test_fuse_int4_groupwise_gemm_idempotent():
    graph = _make_int4_gemm_subgraph(with_intermediate=False)

    graph = PassManager(["fuse_int4_groupwise_gemm"]).optimize(graph, strict=True)
    count_after_first = len(graph)

    graph = PassManager(["fuse_int4_groupwise_gemm"]).optimize(graph, strict=True)
    assert len(graph) == count_after_first
