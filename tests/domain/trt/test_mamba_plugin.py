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
import onnx
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET
from onnxifier.domain.shape_inference import get_shape_inference
from onnxifier.domain.trt.ops.mamba_plugin import (
    causal_conv1d_schema,
    causal_conv1d_shape_inference,
)


def test_node_schema_required_attr():
    graph = make_graph(
        [
            make_node(
                causal_conv1d_schema.name,
                ["x", "w", "b", "state", "context_lengths"],
                ["y", "state_out"],
                "cc",
                domain=causal_conv1d_schema.domain,
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
        [
            make_tensor_value_info("y", 1, [1, 6144, 256]),
            make_tensor_value_info("state_out", 1, [1, 6144, 4]),
        ],
        [
            from_array(np.empty([6144, 1, 4], np.float32), "w"),
            from_array(np.empty([6144], np.float32), "b"),
        ],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[
            ONNXIFIER_OPSET,
            make_operatorsetid(causal_conv1d_schema.domain, 1),
        ],
    )
    onnx.checker.check_model(model, full_check=True)


def test_node_schema_required_io():
    graph = make_graph(
        [
            make_node(
                causal_conv1d_schema.name,
                ["x", "w", "", "", "context_lengths"],
                ["y"],
                "cc",
                domain=causal_conv1d_schema.domain,
                padding=4,
                groups=6144,
            )
        ],
        "test",
        [
            make_tensor_value_info("x", 1, [1, 6144, 256]),
            make_tensor_value_info("context_lengths", TensorProto.INT32, [1]),
        ],
        [make_tensor_value_info("y", 1, [1, 6144, 256])],
        [from_array(np.empty([6144, 1, 4], np.float32), "w")],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[
            ONNXIFIER_OPSET,
            make_operatorsetid(causal_conv1d_schema.domain, 1),
        ],
    )
    onnx.checker.check_model(model, full_check=True)


def test_shape_inference_onnxscript():
    onnxfunc = causal_conv1d_shape_inference.to_function_proto()
    assert "groups" in onnxfunc.attribute
    assert "padding" in onnxfunc.attribute
    out, state = causal_conv1d_shape_inference(
        x=np.zeros([1, 6144, 256], np.float32),
        weight=np.zeros([6144, 1, 4], np.float32),
        bias=None,
        past_conv_state=np.zeros([1, 6144, 4], np.float32),
        context_lengths=np.array([1], np.int32),
        padding=3,
        groups=6144,
        stride=1,
        dilation=1,
    )
    # Simplified shape inference returns x directly, preserving input shape
    assert out.shape == (1, 6144, 256)
    assert state.shape == (1, 6144, 4)

    onnxfunc = get_shape_inference(
        causal_conv1d_schema.domain, causal_conv1d_schema.name
    )
    assert onnxfunc is not None
    assert onnxfunc.domain == causal_conv1d_schema.domain
    assert onnxfunc.name == causal_conv1d_schema.name
    assert len(onnxfunc.input) == len(causal_conv1d_schema.inputs)
    assert len(onnxfunc.output) == len(causal_conv1d_schema.outputs)
