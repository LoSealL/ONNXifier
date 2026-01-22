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

# pylint: disable=missing-function-docstring

import numpy as np
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_type_proto,
    make_value_info,
)
from onnx.numpy_helper import from_array, to_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _build_concat_conv_model():
    weights = np.arange(4 * 8, dtype=np.float32).reshape(4, 8, 1, 1)
    concat = make_node(
        "Concat",
        inputs=["a", "b"],
        outputs=["concat_out"],
        name="Concat",
        axis=1,
    )
    conv = make_node(
        "Conv",
        inputs=["concat_out", "w"],
        outputs=["y"],
        name="Conv",
        kernel_shape=[1, 1],
    )
    graph = make_graph(
        [concat, conv],
        "pad_concat_inputs",
        [
            make_value_info("a", make_tensor_type_proto(1, [1, 3, 4, 4])),
            make_value_info("b", make_tensor_type_proto(1, [1, 5, 4, 4])),
        ],
        [make_value_info("y", make_tensor_type_proto(1, [1, 4, 4, 4]))],
        [from_array(weights, "w")],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    return model, weights


def test_pad_concat_inputs_rewrites_conv_weights():
    model, weights = _build_concat_conv_model()
    runner_ref = Evaluator(model, "onnx")
    inputs = {
        "a": np.random.uniform(-1, 1, size=[1, 3, 4, 4]).astype(np.float32),
        "b": np.random.uniform(-1, 1, size=[1, 5, 4, 4]).astype(np.float32),
    }
    ref_output = runner_ref(["y"], inputs)[0]

    graph = OnnxGraph(model)
    pm = PassManager(["pad_concat_inputs"])
    optimized = pm.optimize(graph, strict=True)

    concat_node = optimized.nodes["Concat"]["pb"]
    assert list(concat_node.input) == [
        "Concat/pad_a_output0",
        "Concat/pad_b_output0",
    ]

    pad_nodes = [
        optimized.nodes["Concat/pad_a"]["pb"],
        optimized.nodes["Concat/pad_b"]["pb"],
    ]
    assert all(node.op_type == "Pad" for node in pad_nodes)

    conv_node = optimized.nodes["Conv"]["pb"]
    assert conv_node.input[1] == "Conv/padded_weight_output_0"

    weight_constant = optimized.nodes["Conv/padded_weight"]["pb"]
    new_weights = to_array(weight_constant.attribute[0].t)
    assert new_weights.shape == (4, 32, 1, 1)

    expected = np.zeros((4, 32, 1, 1), dtype=weights.dtype)
    expected[:, 0:3, ...] = weights[:, 0:3, ...]
    expected[:, 16:21, ...] = weights[:, 3:8, ...]
    np.testing.assert_allclose(new_weights, expected)

    runner_opt = Evaluator(optimized.model, "onnx")
    opt_output = runner_opt(["y"], {k: v.copy() for k, v in inputs.items()})[0]
    np.testing.assert_allclose(opt_output, ref_output, rtol=1e-5)
