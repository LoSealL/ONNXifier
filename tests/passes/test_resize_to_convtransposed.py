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
import pytest
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _build_graph_with_scales(
    mode: str | None = None, ctm: str | None = None, scale: int = 2, ndim: int = 2
):
    attrs: dict = {}
    if mode is not None:
        attrs["mode"] = mode
    if ctm is not None:
        attrs["coordinate_transformation_mode"] = ctm
    resize = make_node(
        "Resize",
        inputs=["x", "", "scales", ""],
        outputs=["y"],
        name="resize",
        **attrs,
    )
    w = 8
    ow = w * scale
    graph = make_graph(
        [resize],
        "graph",
        [make_tensor_value_info("x", TensorProto.FLOAT, (1, 3) + (w,) * ndim)],
        [make_tensor_value_info("y", TensorProto.FLOAT, (1, 3) + (ow,) * ndim)],
        [from_array(np.array([1, 1] + [scale] * ndim, "float32"), name="scales")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, True)
    return model


def _build_graph_with_sizes(
    mode: str | None = None, ctm: str | None = None, scale: int = 2, ndim: int = 2
):
    attrs: dict = {"axes": list(range(2, 2 + ndim))}
    if mode is not None:
        attrs["mode"] = mode
    if ctm is not None:
        attrs["coordinate_transformation_mode"] = ctm
    resize = make_node(
        "Resize",
        inputs=["x", "", "", "sizes"],
        outputs=["y"],
        name="resize",
        **attrs,
    )
    w = 8
    ow = w * scale
    graph = make_graph(
        [resize],
        "graph",
        [make_tensor_value_info("x", TensorProto.FLOAT, (1, 3) + (w,) * ndim)],
        [make_tensor_value_info("y", TensorProto.FLOAT, (1, 3) + (ow,) * ndim)],
        [from_array(np.array([ow] * ndim, "int64"), name="sizes")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, True)
    return model


@pytest.fixture(
    params=[
        # np.random.uniform(0, 255, size=(1, 3, 8)).astype(np.float32),
        np.random.uniform(0, 255, size=(1, 3, 8, 8)).astype(np.float32),
        # np.random.uniform(0, 255, size=(1, 3, 8, 8, 8)).astype(np.float32),
    ]
)
def input_tensors(request):
    """onnxruntime doesn't support interpolate other than 4D tensor"""
    return request.param


@pytest.fixture(params=[2, 3, 4])
def scale_factor(request):
    """Generate scale factors."""
    return request.param


def test_resize_to_convtranspose_nearest_with_scales(input_tensors, scale_factor):
    graph = OnnxGraph(_build_graph_with_scales(mode="nearest", scale=scale_factor))
    runner1 = Evaluator(graph.model)

    pm = PassManager(["resize_to_convtransposed"])
    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model)

    node = graph.nodes["resize"]["pb"]
    assert node.op_type == "ConvTranspose"

    x = input_tensors.copy()
    y = runner1(["y"], {"x": x})[0]
    y_ = runner2(["y"], {"x": x})[0]
    np.testing.assert_allclose(y, y_, rtol=1e-5, atol=1e-5)


def test_resize_to_convtranspose_nearest_with_sizes(input_tensors, scale_factor):
    graph = OnnxGraph(_build_graph_with_sizes(mode="nearest", scale=scale_factor))
    runner1 = Evaluator(graph.model)

    pm = PassManager(["resize_to_convtransposed"])
    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model)

    node = graph.nodes["resize"]["pb"]
    assert node.op_type == "ConvTranspose"

    x = input_tensors
    y = runner1(["y"], {"x": x})[0]
    y_ = runner2(["y"], {"x": x})[0]
    np.testing.assert_allclose(y, y_, rtol=1e-5, atol=1e-5)


@pytest.fixture(
    params=[
        "half_pixel",
        "half_pixel_symmetric",
        "asymmetric",
    ]
)
def ctm_params(request):
    """Generate supported coordinate transformation modes."""
    return request.param


def test_resize_to_convtranspose_linear(input_tensors, ctm_params, scale_factor):
    graph = OnnxGraph(
        _build_graph_with_scales(mode="linear", ctm=ctm_params, scale=scale_factor)
    )
    runner1 = Evaluator(graph.model)

    pm = PassManager(["resize_to_convtransposed"])
    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model)

    node = graph.nodes["resize"]["pb"]
    assert node.op_type == "ConvTranspose"

    # Expect bilinear kernel and pads computed for stride 2
    x = input_tensors
    y = runner1(["y"], {"x": x})[0]
    y_ = runner2(["y"], {"x": x})[0]
    assert y.shape == y_.shape

    slices = [slice(scale_factor - 1, -scale_factor + 1)] * (input_tensors.ndim - 2)
    # psnr
    psnr = 10 * np.log10(255**2 / np.mean((y[..., *slices] - y_[..., *slices]) ** 2))
    assert psnr >= 30, f"PSNR: {psnr:.2f} dB"
