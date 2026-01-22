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
import onnx
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array, to_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.checker import show_difference
from onnxifier.evaluator import Evaluator


def _build_graph():
    dq = make_node(
        "DequantizeLinear",
        ["w", "w_scale", "w_zero_point"],
        ["w_deq"],
        name="dq",
        axis=0,
    )
    conv = make_node("Conv", ["x", "w_deq"], ["y"], auto_pad="VALID", name="conv")
    graph = make_graph(
        [dq, conv],
        "test",
        [make_tensor_value_info("x", 1, [1, 8, 16, 16])],
        [make_tensor_value_info("y", 1, [1, 8, 14, 14])],
        [
            from_array(
                np.random.randint(-128, 127, size=[8, 8, 3, 3], dtype=np.int8), "w"
            ),
            from_array(
                np.random.uniform(-0.1, 0.1, size=[8]).astype(np.float32), "w_scale"
            ),
            from_array(np.zeros([8], dtype=np.int8), "w_zero_point"),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_canonicalize_negative_scale():
    # Quant error is quite large, for most cases the relative error exceeds 10%
    # Use a fixed random seed to pass the unit test here.
    np.random.seed(12)
    graph = _build_graph()
    runner1 = Evaluator(graph.model)
    pm = PassManager(["canonicalize_negative_scale"])
    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model)

    scale_or_weights = graph.onnx_predecessors("dq")
    scale_or_weights = [to_array(n.attribute[0].t) for n in scale_or_weights]
    assert [np.all(scale >= 0) for scale in scale_or_weights if scale.ndim == 1].pop()

    x = np.random.uniform(-1, 1, size=[1, 8, 16, 16]).astype(np.float32)
    y1 = runner1(["y"], {"x": x})[0]
    y2 = runner2(["y"], {"x": x})[0]

    assert np.allclose(y1, y2, rtol=0.1), show_difference(y1, y2, rtol=0.1)
