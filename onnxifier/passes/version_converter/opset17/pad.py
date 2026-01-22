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

from typing import List

import numpy as np
from onnx.onnx_pb import NodeProto

from .... import OnnxGraph
from ...pattern import SingleNodePattern
from ...rewriter import Rewriter
from ...utils import make_constant
from . import OP_CONVERTER


@OP_CONVERTER.register("Pad")
class Pad(Rewriter):
    """Remove axes input."""

    def __init__(self):
        super().__init__(SingleNodePattern("Pad"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        mode = self.get_attribute(node, "mode")
        if mode == "wrap":
            raise ValueError(f"mode {mode} is not supported below opset 18")
        if len(node.input) != 4:
            return
        axes = self.get_value(node.input[3])
        if axes is None:
            raise ValueError(f"axes of {node.name} is not constant")
        pads = self.get_value(node.input[1])
        if pads is None:
            raise ValueError(f"pads of {node.name} is not constant")
        pads = pads.reshape([2, -1])
        input_rank = len(graph.tensor_shape(node.input[0]))
        pads_expand = np.zeros([2, input_rank], np.int64)
        k = 0
        for i in range(input_rank):
            if i in axes:
                pads_expand[:, i] = pads[:, k]
                k += 1
        node.input.pop(3)
        pads_cst = make_constant(f"{node.name}/pads", pads_expand.flatten())
        node.input[1] = pads_cst.output[0]
        self += pads_cst
