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

# pylint: disable=missing-docstring

import tempfile

import numpy as np
import onnx
import pytest
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import ONNXIFIER_OPSET, OnnxGraph, PassManager


def _build_graph(turn_half=False):
    try:
        import torch
        import torchvision as tv
    except ImportError:
        pytest.skip("PyTorch is not available")
    else:
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")

    alexnet = tv.models.alexnet(progress=False)
    inputs = torch.randn(1, 3, 224, 224)
    if turn_half:
        # CPU does not support half precision
        alexnet.half().cuda()
        inputs = inputs.half().cuda()
    with tempfile.TemporaryDirectory() as tmpdir:
        torch.onnx.export(
            alexnet,
            (inputs,),
            tmpdir + "/alexnet.onnx",
            opset_version=17,
            input_names=["images"],
            output_names=["prob"],
        )
        graph = OnnxGraph(onnx.load_model(tmpdir + "/alexnet.onnx"))
    return graph


def _build_graph_with_cast():
    cast0 = make_node("Cast", ["x"], ["y"], to=onnx.TensorProto.FLOAT16)
    cast1 = make_node("Cast", ["y"], ["z"], to=onnx.TensorProto.FLOAT)
    graph = make_graph(
        [cast0, cast1],
        "test_graph",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1000])],
        [make_tensor_value_info("z", onnx.TensorProto.FLOAT, [1, 1000])],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])
    onnx.checker.check_model(model, True)
    return model


def _build_graph_with_resize():
    resize = make_node("Resize", ["x", "", "scales"], ["y"])
    graph = make_graph(
        [resize],
        "test_resize",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT16, [1, 3, 4, 4])],
        [make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [1, 3, 8, 8])],
        initializer=[
            onnx.numpy_helper.from_array(np.array([1, 1, 2, 2], np.float32), "scales")
        ],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def test_half_to_float():
    graph = _build_graph(True)
    pm = PassManager(["half_to_float"])
    graph = pm.optimize(graph, strict=True)
    # check constant
    checked_ops = set()
    for init in graph.initializer:
        assert init.data_type == onnx.TensorProto.FLOAT
        checked_ops.add(init.name)
    assert len(checked_ops) > 0
    # check inputs
    for input_tensor in graph.input:
        assert input_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    # check outputs
    for output_tensor in graph.output:
        assert output_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT


def test_half_to_float_constant():
    graph = _build_graph(True)
    pm = PassManager(["initializer_to_constant", "half_to_float"])
    graph = pm.optimize(graph, strict=True)
    # check constant
    checked_ops = set()
    for node in graph:
        node_pb = graph.nodes[node]["pb"]
        if node_pb.op_type == "Constant":
            assert node_pb.attribute[0].t.data_type == onnx.TensorProto.FLOAT
            checked_ops.add(node)
    assert len(checked_ops) > 0
    # check inputs
    for input_tensor in graph.input:
        assert input_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    # check outputs
    for output_tensor in graph.output:
        assert output_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT


def test_float_to_half():
    graph = _build_graph()
    pm = PassManager(["float_to_half"])
    graph = pm.optimize(graph, strict=True)
    # check constant
    checked_ops = set()
    for init in graph.initializer:
        assert init.data_type == onnx.TensorProto.FLOAT16
        checked_ops.add(init.name)
    assert len(checked_ops) > 0
    # check inputs
    for input_tensor in graph.input:
        assert input_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    # check outputs
    for output_tensor in graph.output:
        assert output_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16


def test_float_to_half_constant():
    graph = _build_graph()
    pm = PassManager(["initializer_to_constant", "float_to_half"])
    graph = pm.optimize(graph, strict=True)
    # check constant
    checked_ops = set()
    for node in graph:
        node_pb = graph.nodes[node]["pb"]
        if node_pb.op_type == "Constant":
            assert node_pb.attribute[0].t.data_type == onnx.TensorProto.FLOAT16
            checked_ops.add(node)
    assert len(checked_ops) > 0
    # check inputs
    for input_tensor in graph.input:
        assert input_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    # check outputs
    for output_tensor in graph.output:
        assert output_tensor.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16


def test_half_to_float_cast():
    graph = OnnxGraph(_build_graph_with_cast())
    pm = PassManager(["half_to_float"])
    graph = pm.optimize(graph, strict=True)
    model = onnx.shape_inference.infer_shapes(graph.model, True, True)
    checked_value = set()
    for value_info in model.graph.value_info:
        assert value_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        checked_value.add(value_info.name)
    assert len(checked_value) > 0


def test_float_to_half_with_resize():
    graph = _build_graph_with_resize()
    pm = PassManager(["float_to_half"])
    graph = pm.optimize(graph, strict=True)
    onnx.checker.check_model(graph.model, True)
