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

import json
from pathlib import Path

import numpy as np
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
    """Build a graph with Conv and MatMul that consume initializer weights."""
    conv = make_node(
        "Conv",
        ["x", "w_conv"],
        ["conv_out"],
        "conv",
        kernel_shape=[3, 3],
        strides=[1, 1],
    )
    matmul = make_node("MatMul", ["conv_flat", "w_matmul"], ["y"], "matmul")
    reshape = make_node(
        "Reshape",
        ["conv_out", "shape"],
        ["conv_flat"],
        "reshape",
    )

    # Conv weight: 8*3*3*3 float32 = 648 elements * 4 bytes = 2592 bytes
    w_conv = np.random.rand(8, 3, 3, 3).astype("float32")
    # MatMul weight: 8*10 float32 = 80 elements * 4 bytes = 320 bytes
    w_matmul = np.random.rand(8, 10).astype("float32")

    graph = make_graph(
        [conv, reshape, matmul],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 8, 8]))],
        [make_value_info("y", make_tensor_type_proto(1, [1, 10]))],
        [
            make_tensor("w_conv", 1, [8, 3, 3, 3], w_conv),
            make_tensor("w_matmul", 1, [8, 10], w_matmul),
            make_tensor("shape", 7, [2], np.array([1, 8], dtype="int64")),
        ],
    )
    return make_model(graph)


def test_inspect_weights_distribution():
    model = _build_graph()
    graph = OnnxGraph(model)
    pm = PassManager(["inspect_weights_distribution"])
    graph = pm.optimize(graph, strict=True)


def test_inspect_weights_distribution_save_json(tmp_path):
    model = _build_graph()
    graph = OnnxGraph(model)
    out_file = str(tmp_path / "weights_report.json")
    pm = PassManager(
        ["inspect_weights_distribution"],
        configs={"inspect_weights_distribution": {"save_path": out_file}},
    )
    graph = pm.optimize(graph, strict=True)

    assert Path(out_file).exists()
    report = json.loads(Path(out_file).read_text(encoding="utf-8"))
    assert "total_weight_bytes" in report
    assert "by_op_type" in report
    assert "Conv" in report["by_op_type"]
    assert "MatMul" in report["by_op_type"]
    # Conv weight: 8*3*3*3*4 = 864
    assert report["by_op_type"]["Conv"]["size_bytes"] == 864
    # MatMul weight: 8*10*4 = 320
    assert report["by_op_type"]["MatMul"]["size_bytes"] == 320
    # Reshape has shape initializer: 2*8 = 16 bytes (int64)
    assert "Reshape" in report["by_op_type"]
    assert report["by_op_type"]["Reshape"]["size_bytes"] == 16
    # total = 864 + 320 + 16 = 1200
    assert report["total_weight_bytes"] == 1200


def test_inspect_weights_distribution_no_weights():
    """Graph with no initializers should not crash."""
    relu = make_node("Relu", ["x"], ["y"], "relu")
    graph = make_graph(
        [relu],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3]))],
        [make_value_info("y", make_tensor_type_proto(1, [1, 3]))],
    )
    model = make_model(graph)
    g = OnnxGraph(model)
    pm = PassManager(["inspect_weights_distribution"])
    g = pm.optimize(g, strict=True)
