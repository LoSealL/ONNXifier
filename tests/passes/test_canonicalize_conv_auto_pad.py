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
from onnx.helper import (
    get_attribute_value,
    make_graph,
    make_model,
    make_node,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager


def _build_graph():
    conv = make_node(
        "Conv", ["x", "w"], ["y"], auto_pad="VALID", pads=[0, 0, 1, 1], name="conv"
    )
    graph = make_graph(
        [conv],
        "test",
        [make_tensor_value_info("x", 1, [1, 8, 16, 16])],
        [make_tensor_value_info("y", 1, [1, 8, 14, 14])],
        [from_array(np.random.randn(8, 8, 3, 3).astype(np.float32), "w")],
    )
    return make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )


def test_canonicalize_conv_autopad():
    model = _build_graph()
    graph = OnnxGraph(model)

    pm = PassManager(["canonicalize_conv_autopad"])
    graph = pm.optimize(graph, strict=True)

    for attr in graph.nodes["conv"]["pb"].attribute:
        if attr.name == "auto_pad":
            assert get_attribute_value(attr).decode() == "NOTSET"
