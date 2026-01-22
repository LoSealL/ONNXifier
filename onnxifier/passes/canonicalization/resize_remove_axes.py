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

from ... import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register("resize_remove_axes", deps=["resize_move_size_to_scale"])
class ResizeRemoveAxesRewriter(Rewriter):
    """Remove axes attribute in Resize operator.

    Since ``axes`` attribute is introduced since opset 18, many backends do not
    support it.

    @Deprecated
    Use ``downgrade_op_version`` instead.
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Resize"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        if axes := self.get_attribute(node, "axes"):  # type: ignore
            input_shape = graph.tensor_shape(node.input[0])
            input_rank = len(input_shape)
            axes = [i if i >= 0 else i + input_rank for i in axes]  # type: ignore
            missing_axes = set(range(input_rank)) - set(axes)
            scales_node = self.get_input_node_or_die(node, 2)
            scales = self.get_value_or_die(scales_node)
            dtype = scales.dtype
            scales_value = {axis: scales[i] for i, axis in enumerate(axes)}
            scales_value.update({axis: 1.0 for axis in missing_axes})
            scales_value = [scales_value[axis] for axis in sorted(scales_value)]
            scales = np.array(scales_value, dtype=dtype)
            new_scales = make_constant(scales_node.name + "/new", scales)
            self.set_attribute(node, "axes", list(range(input_rank)))
            node.input[2] = new_scales.output[0]
            self -= scales_node
            self += new_scales
