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

# pylint: disable=arguments-differ

from collections.abc import Sequence

import numpy as np
from onnx import TensorProto
from onnx.helper import make_node, make_tensor_value_info, tensor_dtype_to_np_dtype
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph, logger
from .. import PASSES
from ..pattern import InputNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


def _find_image_inputs(graph: OnnxGraph) -> list[str]:
    """Find rank-4 NCHW graph inputs whose channel is 1 or 3."""
    names = []
    for name in graph.inputs:
        shape, _ = graph.tensor_info(name)
        if shape is not None and len(shape) == 4 and shape[1] in (1, 3):
            names.append(name)
    return names


def _broadcast(
    values: Sequence[float] | float,
    n: int,
    rank: int,
    np_dtype: np.dtype,
) -> np.ndarray:
    """Expand a scalar or 1-D mean/std to the per-channel broadcast shape."""
    arr = np.asarray(values, dtype=np_dtype)
    if arr.ndim == 0:
        arr = np.full(n, arr.item(), dtype=np_dtype)
    elif arr.ndim == 1 and arr.size in (1, n):
        if arr.size == 1:
            arr = np.full(n, arr.item(), dtype=np_dtype)
    else:
        raise ValueError(
            f"mean/std must be a scalar or have {n} values, got shape {arr.shape}"
        )
    if rank > 1:
        arr = arr.reshape([n] + [1] * (rank - 2))
    return arr


@PASSES.register(name="convert_image_inputs_to_u8", deps=["infer_shape"])
class ConvertImageInputsToU8(Rewriter):
    """Convert image inputs to uint8 with optional preprocessing.

    The input dtype is changed to uint8 and a Cast is inserted so downstream
    nodes still consume the original dtype.

    Args:
        input_names (Sequence[str], optional): names of the image inputs.
            Defaults to rank-4 NCHW inputs with channel 1 or 3.
        mean (Sequence[float] | float, optional): per-channel mean in the
            uint8 value domain, an ``Add(x, -mean)`` is inserted; a scalar or
            length-1 1-D value is expanded to the channel count. Defaults to
            None.
        std (Sequence[float] | float, optional): per-channel std in the uint8
            value domain, a ``Mul(x, 1/std)`` is inserted; a scalar or length-1
            1-D value is expanded to the channel count. Defaults to None.
        packed (bool): the input is packed NHWC uint8 data, a ``Transpose`` is
            inserted so the network still consumes NCHW. Defaults to False.

    Example::

        Before:

            input{f32, NCHW} -> network

        After (mean/std given, packed=True):

            input{u8, NHWC} -> Transpose -> Cast
                -> Add(-mean) -> Mul(1/std) -> network
    """

    def __init__(self):
        super().__init__(pattern=InputNodePattern(match_all=True))

    def rewrite(
        self,
        graph: OnnxGraph,
        nodes: list[NodeProto],
        *_a,
        input_names: Sequence[str] | None = None,
        mean: Sequence[float] | float | None = None,
        std: Sequence[float] | float | None = None,
        packed: bool = False,
        **_kw,
    ):
        if input_names is None:
            input_names = _find_image_inputs(graph)
            if not input_names:
                logger.warning(
                    "no image input found, convert_image_inputs_to_u8 skipped"
                )
                return
        else:
            input_names = list(input_names)
            for name in input_names:
                if name not in graph.inputs:
                    raise ValueError(f"{name} is not a graph input")

        for name in input_names:
            shape, dtype = graph.tensor_info(name)
            assert shape is not None, f"missing shape of input {name}"
            if dtype == TensorProto.UNDEFINED:
                dtype = TensorProto.FLOAT
            if dtype == TensorProto.UINT8:
                logger.warning(f"input {name} is already uint8, skipped")
                continue
            rank = len(shape)
            channel = shape[1] if rank > 1 else shape[0]
            if (mean is not None or std is not None) and not isinstance(channel, int):
                raise ValueError(
                    f"channel of input {name} is dynamic, cannot use mean/std"
                )
            channel = channel if isinstance(channel, int) else 0

            cur = name
            if packed:
                if rank != 4:
                    raise ValueError(
                        f"packed=True requires a rank-4 input, but {name} is {shape}"
                    )
                nhwc = [shape[0], shape[2], shape[3], shape[1]]
                out = f"{name}/transpose_output"
                self += make_node(
                    "Transpose",
                    [cur],
                    [out],
                    name=f"{name}/transpose",
                    perm=[0, 3, 1, 2],
                )
                graph.set_value_info(out, shape, TensorProto.UINT8)
                graph.input[graph.inputs[name]].CopyFrom(
                    make_tensor_value_info(name, TensorProto.UINT8, nhwc)
                )
                cur = out
            else:
                graph.input[graph.inputs[name]].CopyFrom(
                    make_tensor_value_info(name, TensorProto.UINT8, shape)
                )

            out = f"{name}/cast_output"
            self += make_node("Cast", [cur], [out], name=f"{name}/cast", to=dtype)
            graph.set_value_info(out, shape, dtype)
            cur = out

            np_dtype = tensor_dtype_to_np_dtype(dtype)
            if mean is not None:
                self += make_constant(
                    f"{name}/mean", -_broadcast(mean, channel, rank, np_dtype)
                )
                out = f"{name}/add_output"
                self += make_node(
                    "Add", [cur, f"{name}/mean_output_0"], [out], name=f"{name}/add"
                )
                graph.set_value_info(out, shape, dtype)
                cur = out
            if std is not None:
                self += make_constant(
                    f"{name}/std_inv", 1.0 / _broadcast(std, channel, rank, np_dtype)
                )
                out = f"{name}/mul_output"
                self += make_node(
                    "Mul", [cur, f"{name}/std_inv_output_0"], [out], name=f"{name}/mul"
                )
                graph.set_value_info(out, shape, dtype)
                cur = out

            # rewire the original consumers to the last inserted tensor
            for n in graph:
                node = graph.nodes[n]["pb"]
                for i, inp in enumerate(node.input):
                    if inp == name:
                        node.input[i] = cur
