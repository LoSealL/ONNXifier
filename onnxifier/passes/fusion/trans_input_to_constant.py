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

from typing import Optional, Sequence

import numpy as np
from onnx.helper import tensor_dtype_to_np_dtype

from ... import OnnxGraph
from .. import PASSES
from ..utils import make_constant


@PASSES.register()
def trans_input_to_constant(
    graph: OnnxGraph,
    input_name: str | Sequence[str],
    value: Optional[np.ndarray | Sequence[np.ndarray]] = None,
):
    """Consolidate a input to a fixed value as a constant node.

    Args:
        graph (OnnxGraph): onnx graph
        input_name (str | Sequence[str]): input name(s)
        value (np.ndarray | Sequence[np.ndarray], optional): value(s) to be set.
            If None, an all-one array.
    """

    if isinstance(input_name, str):
        input_name = [input_name]
    values: list[None | np.ndarray] = []
    if value is None:
        values = [None for _ in input_name]
    elif not isinstance(value, Sequence):
        values = [value for _ in input_name]
    else:
        values = list(value)  # type: ignore

    assert len(input_name) == len(values)
    for name, v in zip(input_name, values):
        if name not in graph.inputs:
            raise ValueError(f"{name} is not an input of the model")
        shape, ele_type = graph.tensor_info(name)
        if v is None:
            int_shape = graph.static_tensor_shape(name)
            v = np.ones(int_shape, dtype=tensor_dtype_to_np_dtype(ele_type))
        elif shape is not None:
            if len(shape) != v.ndim:
                int_shape = graph.static_tensor_shape(name)
                v = np.broadcast_to(v, int_shape).astype(
                    tensor_dtype_to_np_dtype(ele_type)
                )
            for x, y in zip(shape, v.shape):
                if isinstance(x, int) and x > 0 and x != y:
                    raise ValueError(
                        f"{name} shape {shape} is not compatible with "
                        f"value shape {v.shape}"
                    )
        node = make_constant(name + "/Const", v)
        node.output[0] = name
        graph.add_onnx_node(node)
    return graph
