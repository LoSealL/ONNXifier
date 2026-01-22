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
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)

from onnxifier import OnnxGraph, PassManager


def _build_graph(opset=19, scales=[1, 1, 1, 1, 1]):
    resize = make_node(
        "Resize", ["x", "", "scales"], ["y"], axes=[2, 3, 4], name="resize"
    )
    graph = make_graph(
        [resize],
        "test",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 64, 128, 256])],
        [
            make_tensor_value_info(
                "y", onnx.TensorProto.FLOAT, ["n", "c", "d", "h", "w"]
            )
        ],
        [onnx.numpy_helper.from_array(np.array(scales, np.float32), "scales")],
    )
    model = make_model(graph, opset_imports=[make_operatorsetid("", opset)])
    return OnnxGraph(model)


def test_not_supported_opset():
    graph = _build_graph(opset=17)
    # do nothing if opset is less than 18
    assert graph.opset_version == 17
    pm = PassManager(["remove_resize_batch_and_channel_scales"])
    pm.optimize(graph, strict=True)


def test_remove_resize_batch_and_channel_scales():
    graph = _build_graph(opset=19, scales=(1, 1, 0.5, 0.5, 0.5))
    pm = PassManager(["remove_resize_batch_and_channel_scales", "infer_shape"])
    graph = pm.optimize(graph, strict=True)
    assert graph.tensor_shape("scales/new_output_0") == [3]
    onnx.checker.check_model(graph.model)
    onnx.shape_inference.infer_shapes(graph.model, True, True)

    graph = _build_graph(opset=19, scales=(1, 2, 2, 2, 1))
    pm = PassManager(["remove_resize_batch_and_channel_scales", "infer_shape"])
    graph = pm.optimize(graph, strict=True)
    assert graph.tensor_shape("scales/new_output_0") == [4]
    onnx.checker.check_model(graph.model)
    onnx.shape_inference.infer_shapes(graph.model, True, True)

    graph = _build_graph(opset=19, scales=(2, 2, 1, 1, 1))
    pm = PassManager(["remove_resize_batch_and_channel_scales", "infer_shape"])
    graph = pm.optimize(graph, strict=True)
    assert graph.tensor_shape("scales/new_output_0") == [5]
    onnx.checker.check_model(graph.model)
    onnx.shape_inference.infer_shapes(graph.model, True, True)
