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
import pytest
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.domain.trt.ops.recurrent_plugin import gated_delta_rule_schema


def _make_gdn(custom_domain="trt"):
    graph = make_graph(
        [
            make_node(
                gated_delta_rule_schema.name,
                [
                    "q",
                    "k",
                    "v",
                    "gate",
                    "beta",
                    "past_ssm_state",
                    "context_lengths",
                ],
                ["y", "present_ssm_state"],
                "gdr",
                domain=custom_domain,
                k_dim=64,
                v_dim=64,
                num_v_heads=12,
            )
        ],
        "test",
        [
            make_tensor_value_info("q", 1, [1, 256, 12, 64]),
            make_tensor_value_info("k", 1, [1, 256, 12, 64]),
            make_tensor_value_info("v", 1, [1, 256, 12, 64]),
            make_tensor_value_info("gate", 1, [1, 256, 12]),
            make_tensor_value_info("beta", 1, [1, 256, 12]),
            make_tensor_value_info("past_ssm_state", 1, [1, 12, 64, 64]),
            make_tensor_value_info("context_lengths", TensorProto.INT32, [1]),
        ],
        [
            make_tensor_value_info("y", 1, [1, 256, 12, 64]),
            make_tensor_value_info("present_ssm_state", 1, [1, 12, 64, 64]),
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
    pm = PassManager(["trt_gated_delta_rule_replace"])
    pm.optimize(graph, strict=True)
    if domain != "":
        assert pm.activated[0].num_rewrites == 1
    else:
        assert pm.activated[0].num_rewrites == 0


def test_append_state_and_context_lengths():
    node = make_node(
        gated_delta_rule_schema.name,
        ["q", "k", "v", "gate", "beta"],
        ["y"],
        "gdr",
        domain=gated_delta_rule_schema.domain,
        k_dim=64,
        v_dim=64,
        num_v_heads=12,
    )
    model = make_model(
        make_graph(
            [node],
            "test",
            [
                make_tensor_value_info("q", 1, [1, 256, 12, 64]),
                make_tensor_value_info("k", 1, [1, 256, 12, 64]),
                make_tensor_value_info("v", 1, [1, 256, 12, 64]),
                make_tensor_value_info("gate", 1, [1, 256, 12]),
                make_tensor_value_info("beta", 1, [1, 256, 12]),
                make_tensor_value_info("context_lengths", TensorProto.INT32, [1]),
            ],
            [make_tensor_value_info("y", 1, [1, 256, 12, 64])],
        ),
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[
            ONNXIFIER_OPSET,
            make_operatorsetid(gated_delta_rule_schema.domain, 1),
        ],
    )
    graph = OnnxGraph(model)
    pm = PassManager(["infer_shape", "trt_gated_delta_rule_replace"])
    graph = pm.optimize(graph, strict=True)

    gdr_input = graph.nodes["gdr"]["pb"].input
    gdr_output = graph.nodes["gdr"]["pb"].output
    assert len(gdr_input) == len(gated_delta_rule_schema.inputs)
    assert len(gdr_output) == len(gated_delta_rule_schema.outputs)
    assert "context_lengths" in gdr_input
    assert gdr_input[5].endswith(gated_delta_rule_schema.inputs[5].name)
    assert gdr_input[6] == gated_delta_rule_schema.inputs[6].name
    assert gdr_output[1].endswith(gated_delta_rule_schema.outputs[1].name)
