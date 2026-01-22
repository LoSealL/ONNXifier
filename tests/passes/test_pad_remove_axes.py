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
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import to_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator
from onnxifier.passes.utils import make_constant


def _build_model():
    pads_const = make_constant("Pad/pads", np.array([2, 3, 4, 5], dtype=np.int64))
    axes_const = make_constant("Pad/axes", np.array([1, -1], dtype=np.int64))
    pad = make_node(
        "Pad",
        inputs=["x", pads_const.output[0], "", axes_const.output[0]],
        outputs=["y"],
        name="Pad",
    )
    graph = make_graph(
        [pads_const, axes_const, pad],
        "pad_remove_axes",
        [make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])],
        [make_tensor_value_info("y", TensorProto.FLOAT, [1, "h", "w"])],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    return model


def test_pad_remove_axes_expands_pads_and_removes_axes():
    """Tests for the pad_remove_axes rewriter."""

    model = _build_model()
    runner_ref = Evaluator(model, "onnx")
    x = np.random.uniform(-1, 1, size=[1, 3, 4]).astype(np.float32)
    ref_output = runner_ref(["y"], {"x": x})[0]

    graph = OnnxGraph(model)
    pm = PassManager(["pad_remove_axes"])
    optimized = pm.optimize(graph, strict=True)

    pad_node = optimized.nodes["Pad"]["pb"]
    assert len(pad_node.input) == 3
    assert pad_node.input[2] == ""
    assert pad_node.input[1] == "Pad/expanded_pads_output_0"

    full_pads_node = optimized.nodes["Pad/expanded_pads"]["pb"]
    full_pads = to_array(full_pads_node.attribute[0].t)
    np.testing.assert_array_equal(
        full_pads, np.array([0, 2, 3, 0, 4, 5], dtype=np.int64)
    )

    runner_opt = Evaluator(optimized.model, "onnx")
    opt_output = runner_opt(["y"], {"x": x})[0]
    assert opt_output.shape == (1, 3 + 2 + 4, 4 + 3 + 5)
    np.testing.assert_allclose(opt_output, ref_output)
