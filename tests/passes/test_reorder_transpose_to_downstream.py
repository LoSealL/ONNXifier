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
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET
from onnxifier.evaluator import Evaluator
from onnxifier.graph import OnnxGraph
from onnxifier.pass_manager import PassManager


def _build_graph():
    t0 = make_node("Transpose", ["input"], ["t0"], perm=[0, 1, 3, 2])
    a0 = make_node("Sigmoid", ["t0"], ["a0"])
    s0 = make_node("Slice", ["a0", "b0", "e0", "i0"], ["s0"])
    s1 = make_node("Slice", ["a0", "b1", "e1", "i1"], ["s1"])
    s2 = make_node("Slice", ["a0", "b2", "e2", "i2"], ["s2"])
    m0 = make_node("Mul", ["s1", "x0"], ["m0"])
    m1 = make_node("Mul", ["s2", "x1"], ["m1"])
    add = make_node("Add", ["m0", "x2"], ["add"])
    p0 = make_node("Pow", ["m1", "x3"], ["p0"])
    m2 = make_node("Mul", ["add", "x4"], ["m2"])
    m3 = make_node("Mul", ["p0", "x5"], ["m3"])
    concat = make_node("Concat", ["s0", "m2", "m3"], ["concat"], axis=3)
    reshape = make_node("Reshape", ["concat", "shape"], ["reshape"])
    graph = make_graph(
        [t0, a0, s0, s1, s2, m0, m1, add, p0, m2, m3, concat, reshape],
        "test",
        [make_tensor_value_info("input", 1, [1, 3, 85, 3600])],
        [make_tensor_value_info("reshape", 1, [1, 10800, 85])],
        [
            from_array(np.array([4], np.int64), "b0"),
            from_array(np.array([85], np.int64), "e0"),
            from_array(np.array([3], np.int64), "i0"),
            from_array(np.array([0], np.int64), "b1"),
            from_array(np.array([2], np.int64), "e1"),
            from_array(np.array([3], np.int64), "i1"),
            from_array(np.array([2], np.int64), "b2"),
            from_array(np.array([4], np.int64), "e2"),
            from_array(np.array([3], np.int64), "i2"),
            from_array(np.array(2, np.float32), "x0"),
            from_array(np.array([2], np.float32), "x1"),
            from_array(np.ones([1, 3, 3600, 2], np.float32), "x2"),
            from_array(np.array([2], np.float32), "x3"),
            from_array(np.array(8, np.float32), "x4"),
            from_array(np.ones([1, 3, 3600, 2], np.float32), "x5"),
            from_array(np.array([1, 10800, 85], np.int64), "shape"),
        ],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[ONNXIFIER_OPSET],
    )
    onnx.checker.check_model(model, True)
    return OnnxGraph(model)


def test_reorder_transpose_to_downstream():
    graph = _build_graph()
    runner1 = Evaluator(graph.model)
    pm = PassManager(["reorder_transpose_to_downstream"])
    graph = pm.optimize(graph, strict=True)
    runner2 = Evaluator(graph.model)

    input_feeds = {"input": np.random.rand(1, 3, 85, 3600).astype(np.float32)}
    out1 = runner1(["reshape"], input_feeds)
    out2 = runner2(["reshape"], input_feeds)
    assert out1[0].shape == out2[0].shape
    assert np.allclose(out1[0], out2[0])
