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
    conv1 = make_node(
        "Conv", ["x", "w1"], ["y1"], "conv1", kernel_shape=[2, 2], strides=[2, 2]
    )
    conv2 = make_node(
        "Conv", ["x", "w2"], ["y2"], "conv2", kernel_shape=[2, 2], strides=[2, 2]
    )
    concat = make_node("Concat", ["y1", "y2"], ["z"], "concat", axis=1)
    graph = make_graph(
        [conv1, conv2, concat],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [make_value_info("z", make_tensor_type_proto(1, [1, 12, 128, 128]))],
        [
            make_tensor(
                "w1", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
            make_tensor(
                "w2", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
        ],
    )
    return make_model(graph)


def _build_graph2():
    conv1 = make_node(
        "Conv", ["x", "w1", "b1"], ["y1"], "conv1", kernel_shape=[2, 2], strides=[2, 2]
    )
    conv2 = make_node(
        "Conv", ["x", "w2", "b2"], ["y2"], "conv2", kernel_shape=[2, 2], strides=[2, 2]
    )
    concat = make_node("Concat", ["y1", "y2"], ["z"], "concat", axis=1)
    graph = make_graph(
        [conv1, conv2, concat],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [make_value_info("z", make_tensor_type_proto(1, [1, 12, 128, 128]))],
        [
            make_tensor(
                "w1", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
            make_tensor(
                "w2", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
            make_tensor(
                "b1", 1, [6, 1, 1, 1], np.random.rand(6, 1, 1, 1).astype("float32")
            ),
            make_tensor(
                "b2", 1, [6, 1, 1, 1], np.random.rand(6, 1, 1, 1).astype("float32")
            ),
        ],
    )
    return make_model(graph)


def _build_graph3():
    conv1 = make_node(
        "Conv", ["x", "w1", "b1"], ["y1"], "conv1", kernel_shape=[2, 2], strides=[2, 2]
    )
    conv2 = make_node(
        "Conv", ["x", "w2", "b2"], ["y2"], "conv2", kernel_shape=[2, 2], strides=[2, 2]
    )
    conv3 = make_node(
        "Conv", ["x", "w3", "b3"], ["y3"], "conv3", kernel_shape=[2, 2], strides=[2, 2]
    )
    concat = make_node("Concat", ["y1", "y2", "y3"], ["z"], "concat", axis=1)
    graph = make_graph(
        [conv1, conv3, conv2, concat],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [make_value_info("z", make_tensor_type_proto(1, [1, 12, 128, 128]))],
        [
            make_tensor(
                "w1", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
            make_tensor(
                "w2", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
            make_tensor(
                "w3", 1, [6, 3, 2, 2], np.random.rand(6, 3, 2, 2).astype("float32")
            ),
            make_tensor(
                "b1", 1, [6, 1, 1, 1], np.random.rand(6, 1, 1, 1).astype("float32")
            ),
            make_tensor(
                "b2", 1, [6, 1, 1, 1], np.random.rand(6, 1, 1, 1).astype("float32")
            ),
            make_tensor(
                "b3", 1, [6, 1, 1, 1], np.random.rand(6, 1, 1, 1).astype("float32")
            ),
        ],
    )
    return make_model(graph)


def test_merge_parallel_conv():
    model = _build_graph()
    runner1 = ReferenceEvaluator(model)
    x = np.random.uniform(0, 1, size=[1, 3, 256, 256]).astype(np.float32)
    y1 = runner1.run(None, {"x": x})[0]

    graph = OnnxGraph(model)
    pm = PassManager(["merge_parallel_conv"])
    graph = pm.optimize(graph, strict=True)

    # check output
    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x": x})[0]
    assert np.allclose(y1, y2)

    # check conv number
    conv_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "Conv":
            conv_number += 1
    assert conv_number == 1


def test_merge_parallel_conv2():
    model = _build_graph2()
    runner1 = ReferenceEvaluator(model)
    x = np.random.uniform(0, 1, size=[1, 3, 256, 256]).astype(np.float32)
    y1 = runner1.run(None, {"x": x})[0]

    graph = OnnxGraph(model)
    pm = PassManager(["merge_parallel_conv"])
    graph = pm.optimize(graph, strict=True)

    # check output
    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x": x})[0]
    assert np.allclose(y1, y2)

    # check conv number
    conv_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "Conv":
            conv_number += 1
    assert conv_number == 1


def test_merge_parallel_conv3():
    model = _build_graph3()
    runner1 = ReferenceEvaluator(model)
    x = np.random.uniform(0, 1, size=[1, 3, 256, 256]).astype(np.float32)
    y1 = runner1.run(None, {"x": x})[0]

    graph = OnnxGraph(model)
    pm = PassManager(["merge_parallel_conv"])
    graph = pm.optimize(graph, strict=True)

    # check output
    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x": x})[0]
    assert np.allclose(y1, y2)

    # check conv number
    conv_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "Conv":
            conv_number += 1
    assert conv_number == 1
