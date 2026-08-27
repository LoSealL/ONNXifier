"""
Copyright 2026 The ONNXIFIER Authors

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
import onnx.reference
import pytest
from onnx.helper import (
    make_function,
    make_graph,
    make_model,
    make_node,
    make_tensor_value_info,
)

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager


def _model(subnormal, dtype=np.float16):
    c = np.array(subnormal, dtype=dtype).reshape(1, 1, 2)
    nodes = [
        make_node("Constant", [], ["c"], value=onnx.numpy_helper.from_array(c, "c")),
        make_node("Div", ["x", "c"], ["y"]),
    ]
    graph = make_graph(
        nodes,
        "g",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT16, [1, 1, 2])],
        [make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [1, 1, 2])],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])
    model.ir_version = ONNXIFIER_IR_VERSION
    return model


def _run(model, x):
    ref = onnx.reference.ReferenceEvaluator(model)
    return list(ref.run(None, {"x": x}))[0]


@pytest.mark.parametrize(
    "subnormal,patched",
    [
        ([0.02333, 3.576e-07], True),
        ([1.0, 1.0e-06], True),
        ([0.02333, 0.0417], False),
        ([1.0, 2.0], False),
    ],
)
def test_widen_subnormal_scaling(subnormal, patched):
    graph = OnnxGraph(_model(subnormal))
    got = PassManager(["widen_subnormal_scaling"]).optimize(graph)
    ops = [n.op_type for n in got.model.graph.node]
    if patched:
        assert ops.count("Div") == 1 and ops.count("Mul") == 1
        assert not any(n.op_type == "Cast" for n in got.model.graph.node)
        # the widened divisor constant must be all-normal fp16
        cw = next(
            onnx.numpy_helper.to_array(a.t)
            for n in got.model.graph.node
            if n.op_type == "Constant" and next(iter(n.output), "").endswith("_widen_c")
            for a in n.attribute
            if a.name == "value"
        )
        assert cw.dtype == np.float16
        assert np.all(np.abs(cw) >= 6.103515625e-05)
    else:
        assert ops == ["Constant", "Div"]


def test_widen_subnormal_scaling_numerics():
    # Tiny dividends divided by subnormal divisors stay in fp16 range only if
    # the rescaled division keeps every intermediate representable.
    x = np.array([[[1.2e-06, -3.4e-07]]], dtype=np.float16)
    expected = x.astype(np.float32) / np.array([0.02333, 3.576e-07], np.float32)
    expected = expected.astype(np.float16)
    got_model = PassManager(["widen_subnormal_scaling"]).optimize(
        OnnxGraph(_model([0.02333, 3.576e-07]))
    )
    out = _run(got_model.model, x)
    np.testing.assert_array_equal(out, expected)
    assert not np.isinf(out).any()


def test_widen_subnormal_scaling_zero_dividend():
    # The observed miscompile: 0 divided by a subnormal becomes 0*inf=NaN in
    # TensorRT fp16 fusions; the rescaled rewrite must keep it exactly 0.
    got_model = PassManager(["widen_subnormal_scaling"]).optimize(
        OnnxGraph(_model([0.02333, 3.576e-07]))
    )
    x = np.array([[[0.0, 0.0]]], dtype=np.float16)
    out = _run(got_model.model, x)
    np.testing.assert_array_equal(out, x)
    assert not np.isnan(out).any()


def _nested_model(calls=2):
    """A model whose subnormal Div lives in a function called twice."""
    c = np.array([0.02333, 3.576e-07], dtype=np.float16).reshape(1, 1, 2)
    body = [
        make_node("Constant", [], ["c"], value=onnx.numpy_helper.from_array(c, "c")),
        make_node("Div", ["x", "c"], ["y"]),
    ]
    nodes = [make_node("MyDiv", ["x"], ["t1"]), make_node("MyDiv", ["t1"], ["y"])]
    if calls == 1:
        nodes = [make_node("MyDiv", ["x"], ["y"])]
    graph = make_graph(
        nodes[:calls],
        "g",
        [make_tensor_value_info("x", onnx.TensorProto.FLOAT16, [1, 1, 2])],
        [make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [1, 1, 2])],
    )
    model = make_model(graph, opset_imports=[ONNXIFIER_OPSET])
    model.ir_version = ONNXIFIER_IR_VERSION
    model.functions.extend(
        [
            make_function(
                "",
                "MyDiv",
                ["x"],
                ["y"],
                body,
                opset_imports=[ONNXIFIER_OPSET],
            )
        ]
    )
    return model


@pytest.mark.parametrize("calls", [1, 2])
def test_widen_subnormal_scaling_in_functions(calls):
    # Shared function bodies (multi-call-site) are rewritten by the pass
    # itself, not by PassManager recursion.
    model = _nested_model(calls)
    got = PassManager(["widen_subnormal_scaling"]).optimize(OnnxGraph(model))
    func = got.model.functions[0]
    ops = [n.op_type for n in func.node]
    assert ops.count("Div") == 1 and ops.count("Mul") == 1
    assert not any(n.op_type == "Cast" for n in func.node)
    # the function signature must not change
    assert list(func.input) == ["x"] and list(func.output) == ["y"]


def test_widen_subnormal_scaling_function_numerics():
    # 0 divided by a subnormal stays 0 instead of 0*inf=NaN (the miscompile
    # observed on TensorRT 10.16/11.1 plain path).
    model = _nested_model(calls=2)
    got = PassManager(["widen_subnormal_scaling"]).optimize(OnnxGraph(model))
    func = got.model.functions[0]
    # evaluate the patched function body as a standalone model
    body_model = make_model(
        make_graph(
            list(func.node),
            "body",
            [make_tensor_value_info("x", onnx.TensorProto.FLOAT16, [1, 1, 2])],
            [make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [1, 1, 2])],
        ),
        opset_imports=[ONNXIFIER_OPSET],
    )
    body_model.ir_version = ONNXIFIER_IR_VERSION
    x = np.array([[[0.0, 0.0]]], dtype=np.float16)
    out = _run(body_model, x)
    np.testing.assert_array_equal(out, x)
    assert not np.isnan(out).any()
