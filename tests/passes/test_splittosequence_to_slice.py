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
import pytest
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator
from onnxifier.passes.utils import make_constant


def _build_graph(split: None | np.ndarray, axis: int = 0, keepdims: int = 1):
    nodes = []
    if split is not None:
        split_var = make_constant("split", split)
        nodes.append(split_var)
        split_name = [split_var.output[0]]
    else:
        split_name = []
    nodes.extend(
        [
            make_node(
                "SplitToSequence",
                ["input"] + split_name,
                ["seq"],
                axis=axis,
                keepdims=keepdims,
            ),
            make_node(
                "SequenceAt",
                ["seq", "position"],
                ["output"],
            ),
        ]
    )
    shape: list[int | str] = [2, 3, 4]
    out_shape = shape.copy()
    out_shape[axis] = "C"
    graph = make_graph(
        nodes,
        "test_graph",
        [make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [make_tensor_value_info("output", TensorProto.FLOAT, out_shape)],
        [from_array(np.array(1, np.int64), "position")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    return OnnxGraph(model)


def test_split_to_sequence_to_slice_no_keepdims():
    graph = _build_graph(None, axis=-1, keepdims=0)
    pm = PassManager(["splittosequence_to_slice"])
    graph = pm.optimize(graph, strict=True)

    runner = Evaluator(graph.model, backend="onnxruntime")
    output = runner(["output"], {"input": np.ones([2, 3, 4], np.float32)})[0]
    assert output.shape == (2, 3)


def test_split_to_sequence_to_slice_keepdims():
    graph = _build_graph(None)
    pm = PassManager(["splittosequence_to_slice"])
    graph = pm.optimize(graph, strict=True)

    runner = Evaluator(graph.model, backend="onnxruntime")
    output = runner(["output"], {"input": np.ones([2, 3, 4], np.float32)})[0]
    assert output.shape == (1, 3, 4)


def test_split_to_sequence_to_slice_with_split():
    split = np.array([1, 3], np.int64)
    graph = _build_graph(split, axis=2)
    pm = PassManager(["splittosequence_to_slice"])
    graph = pm.optimize(graph, strict=True)

    runner = Evaluator(graph.model, backend="onnxruntime")
    output = runner(["output"], {"input": np.ones([2, 3, 4], np.float32)})[0]
    assert output.shape == (2, 3, 3)


def test_split_to_sequence_to_slice_with_scalar_split():
    split = np.array(2, np.int64)
    graph = _build_graph(split, axis=2)
    pm = PassManager(["splittosequence_to_slice"])
    graph = pm.optimize(graph, strict=True)

    runner = Evaluator(graph.model, backend="onnxruntime")
    output = runner(["output"], {"input": np.ones([2, 3, 4], np.float32)})[0]
    assert output.shape == (2, 3, 2)


def _build_invalid_graph():
    nodes = []
    nodes.extend(
        [
            make_node(
                "SplitToSequence",
                ["input"],
                ["seq"],
            ),
            make_node(
                "SequenceErase",
                ["seq", "position"],
                ["seq_erased"],
            ),
            make_node(
                "SequenceAt",
                ["seq_erased", "position"],
                ["output"],
            ),
        ]
    )
    shape: list[int | str] = [2, 3, 4]
    out_shape = shape.copy()
    out_shape[0] = "C"
    graph = make_graph(
        nodes,
        "test_graph",
        [make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [make_tensor_value_info("output", TensorProto.FLOAT, out_shape)],
        [from_array(np.array(0, np.int64), "position")],
    )
    model = make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )
    return OnnxGraph(model)


def test_split_to_sequence_to_slice_invalid_graph():
    graph = _build_invalid_graph()
    pm = PassManager(["splittosequence_to_slice"])
    with pytest.raises(NotImplementedError):
        graph = pm.optimize(graph, strict=True)
