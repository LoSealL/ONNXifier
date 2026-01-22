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

from typing import List

import numpy as np
from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from ...logger import warning
from ...passes import PASSES
from ...passes.pattern import SingleNodePattern
from ...passes.rewriter import Rewriter
from ...passes.utils import make_constant


@PASSES.register("pad_remove_axes")
class PadRemoveAxesRewriter(Rewriter):
    """Expand Pad "pads" input and drop the optional "axes" input.

    Note:

        OpenVINO Pad operator does not support "axes" input.

    Before:

        Pad(data, pads, constant_value, axes)

    After:

        Pad(data, expanded_pads, constant_value)
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Pad"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        pad_node = nodes[0]

        if len(pad_node.input) < 4 or not pad_node.input[3]:
            return

        data_input = pad_node.input[0]
        pads_input = pad_node.input[1]
        axes_input = pad_node.input[3]

        try:
            rank = len(graph.tensor_shape(data_input))
        except ValueError as exc:  # tensor shape missing or dynamic rank
            warning(
                f"Pad {pad_node.name} tensor shape unavailable, "
                f"skip removing axes: {exc}"
            )
            return

        pads_value = self.get_value_or_die(pads_input)
        axes_value = self.get_value_or_die(axes_input)

        axes = np.asarray(axes_value).astype(np.int64).tolist()
        if not axes:
            pad_node.input.pop(3)
            return

        axes = [axis + rank if axis < 0 else axis for axis in axes]
        if any(axis < 0 or axis >= rank for axis in axes):
            warning(f"Pad {pad_node.name} has invalid axes {axes}")
            return

        pads_value = np.asarray(pads_value, dtype=np.int64).reshape(-1)
        axis_count = len(axes)
        if pads_value.size != axis_count * 2:
            warning(
                f"Pad {pad_node.name} pads size {pads_value.size} "
                f"mismatches axes length {axis_count}"
            )
            return

        before = pads_value[:axis_count]
        after = pads_value[axis_count:]
        full_pads = np.zeros(rank * 2, dtype=np.int64)
        for idx, axis in enumerate(axes):
            full_pads[axis] = before[idx]
            full_pads[axis + rank] = after[idx]

        pads_const = make_constant(
            f"{pad_node.name}/expanded_pads", full_pads.astype(np.int64)
        )
        pad_node.input[1] = pads_const.output[0]
        pad_node.input.pop(3)

        self += pads_const
