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
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_type_proto,
    make_value_info,
)
from onnx.reference import ReferenceEvaluator

from onnxifier import PassManager
from onnxifier.graph import OnnxGraph


def _build_graph():
    dq1 = make_node("DequantizeLinear", ["x1", "s1", "zp1"], ["y1"], "dq1", axis=0)
    dq2 = make_node("DequantizeLinear", ["x2", "s2", "zp2"], ["y2"], "dq2", axis=0)
    concat = make_node("Concat", ["y1", "y2"], ["z"], "concat", axis=0)
    graph = make_graph(
        [dq1, dq2, concat],
        "graph",
        [
            make_value_info("x1", make_tensor_type_proto(1, [1, 16, 4, 4])),
            make_value_info("x2", make_tensor_type_proto(1, [1, 16, 4, 4])),
        ],
        [make_value_info("z", make_tensor_type_proto(1, [2, 16, 4, 4]))],
        [
            make_tensor("s1", 1, [1], np.random.rand(1).astype("float32")),
            make_tensor("s2", 1, [1], np.random.rand(1).astype("float32")),
            make_tensor(
                "zp1", 3, [1], np.random.randint(0, 127, size=[1], dtype="int8")
            ),
            make_tensor(
                "zp2", 3, [1], np.random.randint(0, 127, size=[1], dtype="int8")
            ),
        ],
    )
    return make_model(graph)


def _build_graph2():
    dq1 = make_node("DequantizeLinear", ["x1", "s1", "zp1"], ["y1"], "dq1", axis=0)
    dq2 = make_node("DequantizeLinear", ["x2", "s2", "zp2"], ["y2"], "dq2", axis=0)
    dq3 = make_node("DequantizeLinear", ["x3", "s3", "zp3"], ["y3"], "dq3", axis=0)
    concat = make_node("Concat", ["y1", "y2", "y3"], ["z"], "concat", axis=0)
    graph = make_graph(
        [dq1, dq2, dq3, concat],
        "graph",
        [
            make_value_info("x1", make_tensor_type_proto(1, [1, 16, 4, 4])),
            make_value_info("x2", make_tensor_type_proto(1, [1, 16, 4, 4])),
            make_value_info("x3", make_tensor_type_proto(1, [1, 16, 4, 4])),
        ],
        [make_value_info("z", make_tensor_type_proto(1, [2, 16, 4, 4]))],
        [
            make_tensor("s1", 1, [1], np.random.rand(1).astype("float32")),
            make_tensor("s2", 1, [1], np.random.rand(1).astype("float32")),
            make_tensor("s3", 1, [1], np.random.rand(1).astype("float32")),
            make_tensor(
                "zp1", 3, [1], np.random.randint(0, 127, size=[1], dtype="int8")
            ),
            make_tensor(
                "zp2", 3, [1], np.random.randint(0, 127, size=[1], dtype="int8")
            ),
            make_tensor(
                "zp3", 3, [1], np.random.randint(0, 127, size=[1], dtype="int8")
            ),
        ],
    )
    return make_model(graph)


def test_swap_concat():
    model = _build_graph()
    runner1 = ReferenceEvaluator(model)
    x1 = np.random.randint(-128, 127, size=[1, 16, 4, 4], dtype="int8")
    x2 = np.random.randint(-128, 127, size=[1, 16, 4, 4], dtype="int8")
    y1 = runner1.run(None, {"x1": x1, "x2": x2})[0]

    graph = OnnxGraph(model)
    pm = PassManager(["swap_concat"])
    graph = pm.optimize(graph, strict=True)

    # check output
    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x1": x1, "x2": x2})[0]
    assert np.allclose(y1, y2)

    # check conv number
    conv_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "DequantizeLinear":
            conv_number += 1
    assert conv_number == 1


def test_swap_concat2():
    model = _build_graph2()
    runner1 = ReferenceEvaluator(model)
    x1 = np.random.randint(-128, 127, size=[1, 16, 4, 4], dtype="int8")
    x2 = np.random.randint(-128, 127, size=[1, 16, 4, 4], dtype="int8")
    x3 = np.random.randint(-128, 127, size=[1, 16, 4, 4], dtype="int8")
    y1 = runner1.run(None, {"x1": x1, "x2": x2, "x3": x3})[0]

    graph = OnnxGraph(model)
    pm = PassManager(["swap_concat"])
    graph = pm.optimize(graph, strict=True)

    # check output
    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x1": x1, "x2": x2, "x3": x3})[0]
    assert np.allclose(y1, y2)

    # check conv number
    conv_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "DequantizeLinear":
            conv_number += 1
    assert conv_number == 1
