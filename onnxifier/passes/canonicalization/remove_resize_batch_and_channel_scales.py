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


@PASSES.register(
    "remove_resize_batch_and_channel_scales",
    deps=["resize_move_size_to_scale"],
)
class RemoveResizeBatchAndChannelRewriter(Rewriter):
    """Remove batch and channel scales in the resize node and add axes
    attribute to spatial sizes."""

    def __init__(self):
        super().__init__(SingleNodePattern("Resize"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        if graph.opset_version < 18:
            return  # axes attribute is added since opset 18
        node = nodes[0]
        input_shape = graph.tensor_shape(node.input[0])
        scales = self.get_input_node_or_die(node, 2)
        scales_value = self.get_value(scales)
        if scales_value is None:
            return
        if len(scales_value) != len(input_shape):
            return
        axes = list(range(len(input_shape)))
        if scales_value[0] == 1:
            axes[0] = None  # type: ignore
        if scales_value[1] == 1:
            axes[1] = None  # type: ignore
        axes = list(filter(lambda x: x is not None, axes))
        self.set_attribute(node, "axes", np.array(axes, dtype=np.int64))
        new_scales = make_constant(scales.name + "/new", scales_value[axes])
        node.input[2] = new_scales.output[0]
        self -= scales
        self += new_scales
