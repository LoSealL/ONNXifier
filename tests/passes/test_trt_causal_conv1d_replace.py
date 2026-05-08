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

# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.domain.trt.ops.mamba_plugin import causal_conv1d_schema


def _make_gdn(custom_domain="trt"):
    graph = make_graph(
        [
            make_node(
                causal_conv1d_schema.name,
                ["x", "w", "b", "state", "context_lengths"],
                ["y"],
                "cc",
                domain=custom_domain,
                padding=4,
                groups=6144,
            )
        ],
        "test",
        [
            make_tensor_value_info("x", 1, [1, 6144, 256]),
            make_tensor_value_info("state", 1, [1, 6144, 4]),
            make_tensor_value_info("context_lengths", TensorProto.INT32, [1]),
        ],
        [make_tensor_value_info("y", 1, [1, 6144, 256])],
        [
            from_array(np.empty([6144, 1, 4], np.float32), "w"),
            from_array(np.empty([6144], np.float32), "b"),
        ],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET, make_operatorsetid(custom_domain, 1)],
    )
    return OnnxGraph(model)


@pytest.mark.parametrize("domain", ["", "trt", "hyper"])
def test_matched_with_any_domain(domain):
    graph = _make_gdn(domain)
    pm = PassManager(["trt_causal_conv1d_replace"])
    pm.optimize(graph, strict=True)
    if domain != "":
        assert pm.activated[0].num_rewrites == 1
    else:
        assert pm.activated[0].num_rewrites == 0


@pytest.mark.parametrize("bias", [False, True])
def test_append_state_and_context_lengths(bias):
    node = make_node(
        causal_conv1d_schema.name,
        ["x", "w"] + (["b"] if bias else []),
        ["y"],
        "cc",
        domain=causal_conv1d_schema.domain,
        padding=4,
        groups=6144,
    )
    model = make_model(
        make_graph(
            [node],
            "test",
            [
                make_tensor_value_info("x", 1, [1, 6144, 256]),
                make_tensor_value_info("context_lengths", TensorProto.INT32, [1]),
            ],
            [make_tensor_value_info("y", 1, [1, 6144, 256])],
            [from_array(np.empty([6144, 1, 4], np.float32), "w")]
            + ([from_array(np.zeros([6144], np.float32), "b")] if bias else []),
        ),
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[
            ONNXIFIER_OPSET,
            make_operatorsetid(causal_conv1d_schema.domain, 1),
        ],
    )
    graph = OnnxGraph(model)
    pm = PassManager(["infer_shape", "trt_causal_conv1d_replace"])
    graph = pm.optimize(graph, strict=True)

    cc_input = graph.nodes["cc"]["pb"].input
    cc_output = graph.nodes["cc"]["pb"].output
    assert len(cc_input) == len(causal_conv1d_schema.inputs)
    assert len(cc_output) == len(causal_conv1d_schema.outputs)
    assert "context_lengths" in cc_input
    assert cc_input[4].endswith(causal_conv1d_schema.inputs[4].name)
