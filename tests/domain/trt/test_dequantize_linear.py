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
import onnx
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET
from onnxifier.domain.shape_inference import get_shape_inference
from onnxifier.domain.trt.ops.dequantize_linear import (
    dequantize_linear_schema,
    from_onnx_dequantize_linear,
    trt_dequantize_linear_shape_infer,
)


def test_dequantize_linear_schema():
    graph = make_graph(
        [
            make_node(
                dequantize_linear_schema.name,
                ["x", "x_scale"],
                ["y"],
                "dq",
                domain=dequantize_linear_schema.domain,
                axis=0,
                block_size=128,
            )
        ],
        "test",
        [
            make_tensor_value_info("x", 1, [4096, 4096]),
            make_tensor_value_info("x_scale", 1, [32, 4096]),
        ],
        [make_tensor_value_info("y", 1, [4096, 4096])],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[
            ONNXIFIER_OPSET,
            make_operatorsetid(dequantize_linear_schema.domain, 1),
        ],
    )
    onnx.checker.check_model(model, full_check=True)


def test_dequantize_linear_schema_with_zero_point():
    graph = make_graph(
        [
            make_node(
                dequantize_linear_schema.name,
                ["x", "x_scale", "x_zero_point"],
                ["y"],
                "dq",
                domain=dequantize_linear_schema.domain,
                axis=0,
                block_size=128,
            )
        ],
        "test",
        [
            make_tensor_value_info("x", 1, [4096, 4096]),
            make_tensor_value_info("x_scale", 1, [32, 4096]),
            make_tensor_value_info("x_zero_point", 1, [32, 4096]),
        ],
        [make_tensor_value_info("y", 1, [4096, 4096])],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[
            ONNXIFIER_OPSET,
            make_operatorsetid(dequantize_linear_schema.domain, 1),
        ],
    )
    onnx.checker.check_model(model, full_check=True)


def test_dequantize_linear_shape_inference():
    out = trt_dequantize_linear_shape_infer(
        x=np.zeros([4096, 4096], np.float32),
        scales=np.zeros([32, 4096], np.float32),
        zero_point=None,
        axis=0,
        block_size=128,
    )
    assert out.shape == (4096, 4096)

    onnxfunc = get_shape_inference(dequantize_linear_schema.domain, "DequantizeLinear")
    assert onnxfunc is not None
    assert onnxfunc.domain == dequantize_linear_schema.domain
    assert onnxfunc.name == "DequantizeLinear"


def test_from_onnx_dequantize_linear():
    op = onnx.NodeProto()
    op.op_type = "DequantizeLinear"
    op.name = "test_dq"
    op.input.extend(["x", "x_scale"])
    op.output.extend(["y"])

    trt_op = from_onnx_dequantize_linear(op)
    assert trt_op.op_type == "DequantizeLinear"
    assert trt_op.domain == dequantize_linear_schema.domain
    assert trt_op.name == "test_dq"
    assert list(trt_op.input) == ["x", "x_scale"]
    assert list(trt_op.output) == ["y"]


def test_from_onnx_dequantize_linear_with_zero_point():
    op = onnx.NodeProto()
    op.op_type = "DequantizeLinear"
    op.name = "test_dq_zp"
    op.input.extend(["x", "x_scale", "x_zero_point"])
    op.output.extend(["y"])

    trt_op = from_onnx_dequantize_linear(op)
    assert trt_op.op_type == "DequantizeLinear"
    assert trt_op.domain == dequantize_linear_schema.domain
    assert list(trt_op.input) == ["x", "x_scale", "x_zero_point"]


def test_from_onnx_dequantize_linear_attributes():
    op = onnx.NodeProto()
    op.op_type = "DequantizeLinear"
    op.name = "test_dq_attrs"
    op.input.extend(["x", "x_scale"])
    op.output.extend(["y"])

    axis_attr = onnx.AttributeProto()
    axis_attr.name = "axis"
    axis_attr.type = onnx.AttributeProto.INT
    axis_attr.i = 0
    op.attribute.append(axis_attr)

    bs_attr = onnx.AttributeProto()
    bs_attr.name = "block_size"
    bs_attr.type = onnx.AttributeProto.INT
    bs_attr.i = 128
    op.attribute.append(bs_attr)

    trt_op = from_onnx_dequantize_linear(op)
    attrs = {a.name: a.i for a in trt_op.attribute}
    assert attrs["axis"] == 0
    assert attrs["block_size"] == 128
