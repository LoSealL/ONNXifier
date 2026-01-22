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

import os
from typing import Dict, Literal, Optional

import numpy as np
import onnx
from onnx.helper import tensor_dtype_to_np_dtype

from .evaluator import Evaluator
from .graph import OnnxGraph


def show_difference(
    a: np.ndarray, b: np.ndarray, atol: float = 1e-8, rtol: Optional[float] = None
):
    """Print the different element from array a and b."""
    if rtol is not None:
        mask = np.abs(1 - b / a) >= rtol
    mask = np.abs(a - b) >= atol
    return f"a={a[mask]}, b={b[mask]}"


def check_accuracy(
    model1: str | os.PathLike | onnx.ModelProto,
    model2: str | os.PathLike | onnx.ModelProto,
    input_maps: Optional[Dict[str, np.ndarray]] = None,
    backend: Literal["onnx", "onnxruntime", "openvino"] = "onnx",
) -> Dict[str, Dict[str, float]]:
    """
    Check the accuracy of two ONNX models.

    Args:
        model1 (onnx.ModelProto): The first ONNX model to be compared.
        model2 (onnx.ModelProto): The second ONNX model to be compared.
        input_maps (Dict[str, numpy.ndarray], optional): The input data to be used for
            the comparison. If not provided, the model will be run with random input
            data.
        backend (Literal["onnx", "onnxruntime", "openvino"], optional): The backend
            to be used for the comparison. Defaults to "onnx".

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the accuracy metrics.
    """

    if not isinstance(model1, onnx.ModelProto):
        model1 = onnx.load_model(model1)
    if not isinstance(model2, onnx.ModelProto):
        model2 = onnx.load_model(model2)
    graph1 = OnnxGraph(model1)
    graph2 = OnnxGraph(model2)

    if input_maps is None:
        input_maps1 = {}
        input_maps2 = {}
        for input_name in graph1.inputs:
            assert input_name in graph2.inputs
            shape = graph1.static_tensor_shape(input_name)
            etype = graph1.tensor_type(input_name)
            dtype = tensor_dtype_to_np_dtype(etype)
            input_maps1[input_name] = np.asarray(np.random.randn(*shape)).astype(dtype)
            # data type may change
            etype = graph2.tensor_type(input_name)
            dtype = tensor_dtype_to_np_dtype(etype)
            input_maps2[input_name] = input_maps1[input_name].astype(dtype)
    else:
        input_maps1 = input_maps
        input_maps2 = input_maps
    output_maps = graph1.outputs
    for output_name in graph2.outputs:
        assert output_name in output_maps

    runner1 = Evaluator(model1, backend)
    runner2 = Evaluator(model2, backend)
    results1 = runner1(list(output_maps), input_maps1)
    results2 = runner2(list(output_maps), input_maps2)

    error_maps: Dict[str, Dict[str, float]] = {}
    for name, x, y in zip(output_maps, results1, results2):
        abs_error = np.abs(x - y)
        rel_mask = np.abs(x) > np.finfo(x.dtype).eps
        rel_error = np.abs(x - y)[rel_mask] / np.abs(x[rel_mask])
        error_maps[name] = {
            "ABS": abs_error.mean(),
            "REL": rel_error.mean(),
        }
    return error_maps
