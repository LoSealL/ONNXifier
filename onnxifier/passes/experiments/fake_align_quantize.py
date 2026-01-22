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


@PASSES.register("fake_align_quantize")
class FakeAlignQuantizeRewriter(Rewriter):
    """Try to align every quantize and dequantize to (zp=0, s=1)."""

    def __init__(self):
        pattern = SingleNodePattern("QuantizeLinear")
        pattern |= SingleNodePattern("DequantizeLinear")
        super().__init__(pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        y_scale = np.ones_like(self.get_value_or_die(node.input[1]))
        y_scale_node = make_constant(f"{node.name}/y_scale", y_scale)
        node.input[1] = y_scale_node.output[0]
        self += y_scale_node
        if len(node.input) >= 2 and node.input[2] != "":
            y_zp = np.zeros_like(self.get_value_or_die(node.input[2]))
            y_zp_node = make_constant(f"{node.name}/y_zp", y_zp)
            node.input[2] = y_zp_node.output[0]
            self += y_zp_node
