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
from onnx.helper import tensor_dtype_to_np_dtype
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph, logger
from ...checker import show_difference
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register("canonicalize_negative_scale")
class CanonicalizeNegativeScaleRewriter(Rewriter):
    """Negative scale leads to quantization overflow on some devices,
    for negative scales, we should flip the sign and check whether quantized
    value exceeds the range.

    Before:

        0.16 (x) / -0.00125 (scale) = -128 (q)

    After:

        0.15875 (x) / 0.00125 (scale) = 127 (q)
    """

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("DequantizeLinear"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        if prev_n := self.get_input_node(node, 0):
            if prev_n.op_type in ("QuantizeLinear",):
                return  # skip activation quantization
        weight = self.get_value(node.input[0])
        if weight is None:
            return  # skip non constant weight
        scale = self.get_value(node.input[1])
        zero_point = self.get_value(node.input[2])
        if scale is None or zero_point is None:
            raise ValueError("scale and zero_point must be constant")
        if scale.ndim < 1 or np.all(scale >= 0) or np.any(zero_point != 0):
            return  # skip tensor-wise scaling, asymmetric scaling

        axis = self.get_attribute(node, "axis")
        if axis is None:
            axis = 0
        else:
            assert isinstance(axis, (int, float, str))
            axis = int(axis)
        neg_mask = scale < 0
        logger.debug(f"======={node.name}=======")
        logger.debug(f"neg scales {scale[neg_mask]} on channels {np.nonzero(neg_mask)}")
        shape = np.ones([weight.ndim], np.int64)
        shape[axis] = -1
        dq_weight = (weight.astype(np.float32)) * scale.reshape(shape)
        scale = scale.copy()
        scale[neg_mask] = -scale[neg_mask]

        etype = graph.tensor_type(node.input[0])
        dtype = tensor_dtype_to_np_dtype(etype)
        dinfo = np.iinfo(dtype)
        reweight = np.round(dq_weight / scale.reshape(shape)).clip(dinfo.min, dinfo.max)
        reweight = reweight.astype(dtype)
        weight_mask = weight != reweight
        diff = show_difference(
            np.abs(weight.astype("int32")), np.abs(reweight.astype("int32")), rtol=0.01
        )
        logger.debug(f"reweight: {weight[weight_mask]} -> {reweight[weight_mask]}")
        logger.debug(f"diff: {diff}")
        weight_node = make_constant(f"{node.name}/new_weight", reweight)
        scale_node = make_constant(f"{node.name}/new_scale", scale)
        node.input[0] = weight_node.output[0]
        node.input[1] = scale_node.output[0]
        self += [weight_node, scale_node]
