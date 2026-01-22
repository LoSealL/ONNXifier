"""
Copyright (C) 2024 The ONNXIFIER Authors.

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
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array, to_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager


def _build_graph():
    dq = make_node(
        "DequantizeLinear",
        ["x_q", "x_scale", "x_zero_point"],
        ["x_fq"],
        name="dq",
        axis=1,
    )
    q = make_node(
        "QuantizeLinear",
        ["x", "x_scale", "x_zero_point"],
        ["x_q"],
        name="q",
        axis=1,
    )
    graph = make_graph(
        [q, dq],
        "test",
        [make_tensor_value_info("x", 1, [1, 8, 16, 16])],
        [make_tensor_value_info("x_fq", 1, [1, 8, 16, 16])],
        [
            from_array(
                np.random.uniform(0.001, 0.1, size=[8]).astype(np.float32), "x_scale"
            ),
            from_array(
                np.random.randint(-2, 1, size=[8], dtype=np.int8), "x_zero_point"
            ),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_canonicalize_negative_zero_point():
    graph = _build_graph()
    pm = PassManager(["canonicalize_negative_zero_point"])
    graph = pm.optimize(graph, strict=True)

    zp_cst = graph.nodes["dq/zero_point"]["pb"]
    zero_point = to_array(zp_cst.attribute[0].t)
    assert np.all(zero_point >= 0)
