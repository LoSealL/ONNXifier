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
from onnx.reference import ReferenceEvaluator

from onnxifier import PassManager
from onnxifier.graph import OnnxGraph


def _build_graph():
    conv = make_node(
        "Conv", ["x", "w"], ["y"], "conv", kernel_shape=[2, 2], strides=[2, 2]
    )
    graph = make_graph(
        [conv],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 256, 256]))],
        [make_value_info("y", make_tensor_type_proto(1, [1, 12, 128, 128]))],
        [
            make_tensor(
                "w", 1, [12, 3, 2, 2], np.random.rand(12, 3, 2, 2).astype("float32")
            )
        ],
    )
    return make_model(graph)


def _build_qdq_graph():
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
            numpy_helper.from_array(np.random.randint(0, 256, [12], "uint8"), "z2"),
            numpy_helper.from_array(
                np.random.randint(0, 256, [12, 3, 2, 2], "uint8"), "qw"
            ),
        ],
    )
    return make_model(graph)


def test_expand_conv_channel():
    model = _build_graph()
    runner1 = ReferenceEvaluator(model)
    x = np.random.uniform(0, 1, size=[1, 3, 256, 256]).astype(np.float32)
    y1 = runner1.run(None, {"x": x})[0]

    graph = OnnxGraph(model)
    pm = PassManager(["expand_conv_channel"])
    graph = pm.optimize(graph, strict=True)

    runner2 = ReferenceEvaluator(graph.model)
    x_expand = np.random.uniform(0, 1, size=[1, 4, 256, 256]).astype(np.float32)
    x_expand[:, :3, :, :] = x
    y2 = runner2.run(None, {"x_expand": x_expand})[0]
    assert np.allclose(y1, y2)


def test_expand_qconv_channel():
    model = _build_qdq_graph()
    runner1 = ReferenceEvaluator(model)
    x = np.random.uniform(0, 1, size=[1, 3, 256, 256]).astype(np.float32)
    y1 = runner1.run(None, {"inp": x})[0]

    graph = OnnxGraph(model)
    pm = PassManager(["expand_qconv_channel"])
    graph = pm.optimize(graph, strict=True)

    runner2 = ReferenceEvaluator(graph.model)
    x_expand = np.random.uniform(0, 1, size=[1, 4, 256, 256]).astype(np.float32)
    x_expand[:, :3, :, :] = x
    y2 = runner2.run(None, {"inp_expand": x_expand})[0]
    assert np.allclose(y1, y2)
