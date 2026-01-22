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

# pylint: disable=arguments-differ
from typing import List

import numpy as np
from onnx import TensorProto
from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register("canonicalize_uint8_weights")
class CanonicalizeUint8WeightsRewriter(Rewriter):
    """Some devices (E.g. SM8650) do not support asymmetric weight quantization.
    If weights are quantized to uint8, this pass convert it to symmetric quantization.
    """

    def __init__(self):
        conv = SingleNodePattern("Conv")
        conv |= SingleNodePattern("ConvTranspose")
        conv |= SingleNodePattern("MatMul")
        pattern = GraphPattern()
        pattern.add_edge(SingleNodePattern("DequantizeLinear"), conv)
        super().__init__(pattern=pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        dq_node, conv_node = nodes
        if self.get_input_node_or_die(conv_node, 1) != dq_node:
            # just to match the dequantize of weights, not the activations
            return
        x_dtype = graph.tensor_type(dq_node.input[0])
        if x_dtype != TensorProto.UINT8:
            return

        x = self.get_value_or_die(dq_node.input[0])
        x_scale = self.get_value_or_die(dq_node.input[1])
        x_zp = self.get_value_or_die(dq_node.input[2])
        if conv_node.op_type == "ConvTranspose":
            channel_axis = 1
        else:
            channel_axis = 0
        expand_axis = np.arange(x.ndim).tolist()
        expand_axis.pop(channel_axis)
        x_scale = np.expand_dims(x_scale, axis=expand_axis)
        x_zp = np.expand_dims(x_zp, axis=expand_axis)
        fx = (x.astype(np.float32) - x_zp) * x_scale
        x_max = np.abs(fx)
        for i in expand_axis:
            x_max = np.max(x_max, axis=i, keepdims=True)
        x_max = np.squeeze(x_max)
        x_new_scale = (x_max / np.iinfo(np.int8).max).astype(np.float32)
        x_new_scale[x_new_scale == 0] = 1
        x_new_zp = np.zeros_like(x_zp).reshape([-1]).astype(np.int8)
        x_new = (
            (fx / np.expand_dims(x_new_scale, axis=expand_axis))
            .round()
            .clip(-128, 127)
            .astype(np.int8)
        )
        x_cst = make_constant(f"{dq_node.name}/x", x_new)
        scale_cst = make_constant(f"{dq_node.name}/x_scale", x_new_scale)
        zp_cst = make_constant(f"{dq_node.name}/x_zp", x_new_zp)

        dq_node.input[0] = x_cst.output[0]
        dq_node.input[1] = scale_cst.output[0]
        dq_node.input[2] = zp_cst.output[0]
        self += [x_cst, scale_cst, zp_cst]
