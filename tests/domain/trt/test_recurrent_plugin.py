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
import onnx
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET
from onnxifier.domain.trt.ops.recurrent_plugin import gated_delta_rule_schema


def test_node_schema_required_attr():
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
                domain=gated_delta_rule_schema.domain,
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
        opset_imports=[
            ONNXIFIER_OPSET,
            make_operatorsetid(gated_delta_rule_schema.domain, 1),
        ],
    )
    onnx.checker.check_model(model, full_check=True)


def test_node_schema_required_io():
    graph = make_graph(
        [
            make_node(
                gated_delta_rule_schema.name,
                ["q", "k", "v", "gate", "beta", "", "context_lengths"],
                ["y"],
                "gdr",
                domain=gated_delta_rule_schema.domain,
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
            make_tensor_value_info("context_lengths", TensorProto.INT32, [1]),
        ],
        [make_tensor_value_info("y", 1, [1, 256, 12, 64])],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[
            ONNXIFIER_OPSET,
            make_operatorsetid(gated_delta_rule_schema.domain, 1),
        ],
    )
    onnx.checker.check_model(model, full_check=True)
