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

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _build_avgpool(
    kernel: tuple[int, ...] = (3, 3),
    strides: tuple[int, ...] = (1, 1),
    pads: tuple[int, ...] | None = None,
    auto_pad: str | None = None,
    count_include_pad: int = 1,
    ndim: int = 2,
):
    attrs: dict = {"kernel_shape": list(kernel), "strides": list(strides)}
    if pads is not None:
        attrs["pads"] = list(pads)
    if auto_pad is not None:
        attrs["auto_pad"] = auto_pad
    if count_include_pad is not None:
        attrs["count_include_pad"] = int(count_include_pad)

    node = make_node("AveragePool", inputs=["x"], outputs=["y"], name="avg", **attrs)
    w = 8
    # compute output spatial sizes assuming ceil_mode=0 and auto_pad NOTSET
    if pads is None:
        pads_list = [0] * (2 * ndim)
    else:
        pads_list = list(pads)
    out_spatial = []
    for d in range(ndim):
        in_d = w
        k = kernel[d]
        s = strides[d]
        p0 = pads_list[d]
        p1 = pads_list[d + ndim]
        out_d = (in_d + p0 + p1 - k) // s + 1
        out_spatial.append(int(out_d))
    graph = make_graph(
        [node],
        "g",
        [make_tensor_value_info("x", TensorProto.FLOAT, (1, 3) + (w,) * ndim)],
        [make_tensor_value_info("y", TensorProto.FLOAT, (1, 3) + tuple(out_spatial))],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model, True)
    return model


@pytest.mark.parametrize("kernel,strides", [((3, 3), (1, 1)), ((2, 2), (2, 2))])
def test_avgpool_to_conv_basic(kernel, strides):
    model = _build_avgpool(kernel=kernel, strides=strides, count_include_pad=1)
    graph = OnnxGraph(model)
    runner1 = Evaluator(graph.model)

    pm = PassManager(["avgpool_to_conv"])
    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model)

    node = graph.nodes["avg"]["pb"]
    assert node.op_type == "Conv"

    x = np.random.randn(1, 3, 8, 8).astype(np.float32)
    y1 = runner1(["y"], {"x": x})[0]
    y2 = runner2(["y"], {"x": x})[0]
    np.testing.assert_allclose(y1, y2, rtol=1e-5, atol=1e-5)


def test_avgpool_to_conv_with_pads_include():
    # count_include_pad=1 with non-zero pads should be supported
    model = _build_avgpool(
        kernel=(3, 3), strides=(2, 2), pads=(1, 1, 1, 1), count_include_pad=1
    )
    graph = OnnxGraph(model)
    runner1 = Evaluator(graph.model)

    pm = PassManager(["avgpool_to_conv"])
    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model)

    node = graph.nodes["avg"]["pb"]
    assert node.op_type == "Conv"

    x = np.random.randn(1, 3, 8, 8).astype(np.float32)
    y1 = runner1(["y"], {"x": x})[0]
    y2 = runner2(["y"], {"x": x})[0]
    np.testing.assert_allclose(y1, y2, rtol=1e-5, atol=1e-5)


def test_avgpool_to_conv_skip_when_exclude_pad_with_nonzero_pad():
    # count_include_pad=0 with pads!=0 cannot be converted; pass should not fire
    model = _build_avgpool(
        kernel=(3, 3), strides=(1, 1), pads=(1, 1, 1, 1), count_include_pad=0
    )
    graph = OnnxGraph(model)

    pm = PassManager(["avgpool_to_conv"])
    graph = pm.optimize(graph, strict=True)

    node = graph.nodes["avg"]["pb"]
    assert node.op_type == "AveragePool"
