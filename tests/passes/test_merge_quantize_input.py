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
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _build_graph():
    # Create a graph with two inputs and one output
    # The first input is quantized with scale 0.5 and zero point 0
    # The second input is not quantized
    qnode0 = make_node(
        "QuantizeLinear",
        ["input0", "scale0", "zero_point0"],
        ["quantized0"],
        "quantize0",
    )
    dqnode0 = make_node(
        "DequantizeLinear",
        ["quantized0", "scale1", "zero_point1"],
        ["output0"],
        "dequantize0",
    )
    add = make_node("Add", ["output0", "input1"], ["sum"], "add")
    graph = make_graph(
        [qnode0, dqnode0, add],
        "graph",
        [
            make_value_info("input0", make_tensor_type_proto(1, [1, 3, 224, 224])),
            make_value_info("input1", make_tensor_type_proto(1, [1, 3, 224, 224])),
        ],
        [make_value_info("sum", make_tensor_type_proto(1, [1, 3, 224, 224]))],
        [
            make_tensor("scale0", 1, [], [1.0]),
            make_tensor("zero_point0", TensorProto.UINT8, [], [0]),
            make_tensor("scale1", 1, [], [1.0]),
            make_tensor("zero_point1", TensorProto.UINT8, [], [0]),
        ],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    return model


def test_merge_quantize_input():
    graph = OnnxGraph(_build_graph())
    pm = PassManager(["initializer_to_constant", "infer_shape", "merge_quantize_input"])
    graph = pm.optimize(graph, strict=True)
    assert graph.tensor_type("input0") == TensorProto.UINT8
    assert graph.tensor_type("input1") == TensorProto.FLOAT

    ori_eval = Evaluator(_build_graph(), "OnnxRuntime")
    opt_eval = Evaluator(graph.model, "OnnxRuntime")

    input0 = np.random.uniform(0, 255, size=[1, 3, 224, 224]).astype(np.float32)
    input1 = np.random.uniform(-128, 127, size=[1, 3, 224, 224]).astype(np.float32)
    qinput0 = np.round(input0).clip(0, 255).astype(np.uint8)
    ori_res = ori_eval(["sum"], {"input0": input0, "input1": input1})[0]
    opt_res = opt_eval(["sum"], {"input0": qinput0, "input1": input1})[0]
    error = np.abs(ori_res - opt_res).max()
    assert np.allclose(ori_res, opt_res), f"Error: {error}"
