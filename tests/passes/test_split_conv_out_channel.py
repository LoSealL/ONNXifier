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
import onnx
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _build_conv():
    conv = make_node(
        "Conv",
        ["X", "W", "B"],
        ["Y"],
        "conv",
        group=1,
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    graph = make_graph(
        [conv],
        "test",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 64, 16, 16])],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 1024, 16, 16])],
        [
            from_array(np.random.rand(1024, 64, 3, 3).astype(np.float32), "W"),
            from_array(np.random.rand(1024).astype(np.float32), "B"),
        ],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def _build_dwconv():
    conv = make_node(
        "Conv",
        ["X", "W", "B"],
        ["Y"],
        "conv",
        group=1024,
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    graph = make_graph(
        [conv],
        "test",
        [make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1024, 16, 16])],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 1024, 16, 16])],
        [
            from_array(np.random.rand(1024, 1, 3, 3).astype(np.float32), "W"),
            from_array(np.random.rand(1024).astype(np.float32), "B"),
        ],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def test_split_conv_out_channel():
    graph = _build_conv()
    run1 = Evaluator(graph.model, "OnnxRuntime")
    pm = PassManager(["split_conv_out_channel"])
    graph = pm.optimize(graph, True)
    run2 = Evaluator(graph.model, "OnnxRuntime")
    assert 8 == len(
        list(
            filter(
                lambda n: n["pb"].op_type == "Conv",
                graph.nodes.values(),
            )
        )
    )

    x = np.random.rand(1, 64, 16, 16).astype(np.float32)
    y1 = run1(["Y"], {"X": x})[0]
    y2 = run2(["Y"], {"X": x})[0]
    error = np.abs(y1 - y2).max()
    assert np.allclose(y1, y2), f"Error: {error}"


def test_split_dwconv_out_channel():
    graph = _build_dwconv()
    run1 = Evaluator(graph.model, "OnnxRuntime")
    pm = PassManager(["split_conv_out_channel"])
    graph = pm.optimize(graph, True)
    run2 = Evaluator(graph.model, "OnnxRuntime")
    assert 8 == len(
        list(
            filter(
                lambda n: n["pb"].op_type == "Conv",
                graph.nodes.values(),
            )
        )
    )

    x = np.random.rand(1, 1024, 16, 16).astype(np.float32)
    y1 = run1(["Y"], {"X": x})[0]
    y2 = run2(["Y"], {"X": x})[0]
    error = np.abs(y1 - y2).max()
    assert np.allclose(y1, y2), f"Error: {error}"
