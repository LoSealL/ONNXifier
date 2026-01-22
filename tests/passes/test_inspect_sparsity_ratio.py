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
from onnx import numpy_helper
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier import PassManager
from onnxifier.graph import OnnxGraph


def _build_graph():
    conv = make_node(
        "Conv", ["x", "w"], ["y"], "conv", kernel_shape=[2, 2], strides=[2, 2]
    )
    # build 1/4 sparsity ratio weight
    weight = np.random.rand(12, 3, 2, 2).astype("float32") + 1e-6
    weight[:, :, 0, 1] = 0

    graph = make_graph(
        [conv],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [make_value_info("y", make_tensor_type_proto(1, [1, 12, 128, 128]))],
        [make_tensor("w", 1, [12, 3, 2, 2], weight)],
    )
    return make_model(graph)


def _build_qdq_graph(weight: np.ndarray, zero_points: np.ndarray):
    conv = make_node(
        "Conv", ["x", "w"], ["y"], "conv", kernel_shape=[2, 2], strides=[2, 2]
    )
    qlinear = make_node("QuantizeLinear", ["inp", "s0", "z0"], ["q"], "qlinear")
    dqlinear = make_node("DequantizeLinear", ["q", "s1", "z1"], ["x"], "dqlinear")
    wdqlinear = make_node(
        "DequantizeLinear", ["qw", "s2", "z2"], ["w"], "wdqlinear", axis=0
    )
    graph = make_graph(
        [qlinear, dqlinear, wdqlinear, conv],
        "graph",
        [make_value_info("inp", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [
            make_value_info(
                "y",
                make_tensor_type_proto(1, [1, 12, 128, 128]),
            )
        ],
        [
            numpy_helper.from_array(np.ones([1], "float32"), "s0"),
            numpy_helper.from_array(np.zeros([1], "uint8"), "z0"),
            numpy_helper.from_array(np.ones([1], "float32"), "s1"),
            numpy_helper.from_array(np.zeros([1], "uint8"), "z1"),
            numpy_helper.from_array(np.ones([12], "float32"), "s2"),
            numpy_helper.from_array(zero_points, "z2"),
            numpy_helper.from_array(weight, "qw"),
        ],
    )
    return make_model(graph)


def test_inspect_sparsity_ratio_float():
    model = _build_graph()
    graph = OnnxGraph(model)
    pm = PassManager(["inspect_sparsity_ratio"])
    # sparsity ratio should be 25%
    graph = pm.optimize(graph, strict=True)


def test_inspect_sparsity_ratio_uint8():
    zp = np.array(range(12), dtype="uint8")
    weight = np.tile(zp.reshape([12, 1, 1, 1]), [3, 2, 2])
    model = _build_qdq_graph(weight=weight, zero_points=zp)
    graph = OnnxGraph(model)
    pm = PassManager(["inspect_sparsity_ratio"])
    # sparsity ratio should be 100%
    graph = pm.optimize(graph, strict=True)


def test_inspect_sparsity_ratio_uint8_2():
    zp = np.array([0] * 12, dtype="uint8")
    weight = np.random.randint(1, 255, [12, 3, 2, 2], dtype="uint8")
    weight[:, 0] = 0
    model = _build_qdq_graph(weight=weight, zero_points=zp)
    graph = OnnxGraph(model)
    pm = PassManager(["inspect_sparsity_ratio"])
    # sparsity ratio should be 33.33%
    graph = pm.optimize(graph, strict=True)
