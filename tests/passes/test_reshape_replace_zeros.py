"""
Copyright (C) 2025 The ONNXIFIER Authors.

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
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.passes.utils import evaluate_on_node


def test_reshape_replace_zeros():
    reshape = make_node("Reshape", ["x", "shape"], ["y"], name="reshape")
    g = make_graph(
        [reshape],
        "reshape",
        [make_tensor_value_info("x", 1, [1, 8, 128])],
        [make_tensor_value_info("y", 1, [1, 8, 8, 16])],
        [from_array(np.array([1, 0, 8, -1], np.int64), "shape")],
    )
    model = make_model(
        g, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)

    pm = PassManager(["reshape_replace_zeros"])
    graph = pm.optimize(OnnxGraph(model), strict=True)

    shape = evaluate_on_node(
        graph, graph.onnx_predecessors(graph.nodes["reshape"]["pb"])[0]
    )
    assert shape is not None
    assert shape.tolist() == [1, 8, 8, 16]


def test_reshape_replace_zeros_no_change_dynamic_shape():
    reshape = make_node("Reshape", ["x", "shape"], ["y"], name="reshape")
    g = make_graph(
        [reshape],
        "reshape",
        [make_tensor_value_info("x", 1, [1, "C", 128])],
        [make_tensor_value_info("y", 1, [1, "C", 8, 16])],
        [from_array(np.array([1, 0, 8, -1], np.int64), "shape")],
    )
    model = make_model(
        g, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)

    pm = PassManager(["reshape_replace_zeros"])
    # nothing changes with no error
    pm.optimize(OnnxGraph(model), strict=True)
