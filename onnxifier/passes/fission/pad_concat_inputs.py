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

from typing import List, Optional

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from ...logger import warning
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register("pad_concat_inputs")
class PadConcatInputsRewriter(Rewriter):
    """
    Pad channels of each input of Concat operator to the multiple of `P`.
    This pass is to match the memory layout to omit the layout change during
    concatenation.

    Before:

        Concat(a, b, c)  # a[C1], b[C2], c[C3]
          |
          OP

    After:

        Concat(Pad(a), Pad(b), Pad(c))  # Pad(a)[(C1+P-1)//P*P]
          |
          OP' # Downstream operator should be updated accordingly
    """

    SUPPORT_OP_TYPES = ("Conv",)

    def __init__(self, multiplier: int = 16):
        pattern = GraphPattern()
        pattern.add_edge(SingleNodePattern("Concat"), SingleNodePattern())
        super().__init__(pattern)
        self.p = multiplier

    # pylint: disable=arguments-differ
    def rewrite(
        self, graph: OnnxGraph, nodes: List[NodeProto], multiplier: Optional[int] = None
    ):
        if multiplier:
            self.p = multiplier
        concat, op = nodes
        if op.op_type not in PadConcatInputsRewriter.SUPPORT_OP_TYPES:
            return
        axis: int = self.get_attribute(concat, "axis", 0)  # type: ignore
        pad: list[tuple[int, int]] = []  # (before, after) for each dim
        for i, inp in enumerate(concat.input):
            shape = graph.tensor_shape(inp)
            channel = shape[axis]
            if not isinstance(channel, int):
                warning(f"Concat {concat.name} input {inp} has dynamic shape: {shape}")
                return
            padded_channel = (channel + self.p - 1) // self.p * self.p
            pad.append((channel, padded_channel))
            if channel != padded_channel:
                name = f"{concat.name}/pad_{inp}"
                pads = make_constant(
                    f"{name}/pads", np.array([0, padded_channel - channel], np.int64)
                )
                axes = make_constant(f"{name}/axes", np.array([axis], np.int64))
                pad_node = make_node(
                    "Pad",
                    [inp, pads.output[0], "", axes.output[0]],
                    [f"{name}_output0"],
                    name=name,
                )
                concat.input[i] = pad_node.output[0]
                self += [pads, axes, pad_node]
        if op.op_type == "Conv":
            if axis != 1:
                raise ValueError("For downstream Conv, Concat axis must be 1.")
            return self.rewrite_conv(op, pad)
        raise RuntimeError(f"Unsupported downstream operator: {op.op_type}")

    def rewrite_conv(self, conv: NodeProto, pad: list[tuple[int, int]]):
        """Rewrite conv weights according to the padded input channels."""
        group: int = self.get_attribute(conv, "group", 1)  # type: ignore
        weights = self.get_value_or_die(conv.input[1])
        if group != 1:
            raise ValueError("Concat rewrite does not support grouped conv.")

        weight_shape = list(weights.shape)
        weight_shape[1] = sum([p[1] for p in pad])
        new_weights = np.zeros(weight_shape, dtype=weights.dtype)
        off = 0
        for i, (before, after) in enumerate(pad):
            new_weights[:, off : off + before, ...] = weights[
                :, sum(p[0] for p in pad[:i]) : sum(p[0] for p in pad[: i + 1]), ...
            ]
            off += after
        weight_node = make_constant(f"{conv.name}/padded_weight", new_weights)
        conv.input[1] = weight_node.output[0]
        self += weight_node
