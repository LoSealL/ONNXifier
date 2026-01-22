"""
Copyright (C) 2024-2025 The ONNXIFIER Authors.

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

import numpy as np
from onnx.helper import make_node, tensor_dtype_to_np_dtype
from onnx.onnx_pb import NodeProto

from .. import OnnxGraph, make_constant
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register("group_normalization")
class GroupNormalization(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/normalization/group-normalization-12.html

    https://onnx.ai/onnx/operators/onnx__GroupNormalization.html

    Note:

        The definition of scale and bias in onnx diverses from pytorch and openvino.
        We need to add extra mul and add to perform a per-channel affine.
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        etype = graph.tensor_type(ori_node.input[0])
        dtype = tensor_dtype_to_np_dtype(etype)
        num_groups = int(self.get_attribute(ori_node, "num_groups"))  # type: ignore
        epsilon = float(self.get_attribute(ori_node, "epsilon"))  # type: ignore
        ones = make_constant(f"{ori_node.name}/scale", np.ones([num_groups], dtype))
        zeros = make_constant(f"{ori_node.name}/bias", np.zeros([num_groups], dtype))
        gn = make_node(
            "GroupNormalization",
            inputs=[ori_node.input[0], ones.output[0], zeros.output[0]],
            outputs=ori_node.output,
            name=ori_node.name,
            epsilon=float(epsilon),
            num_groups=int(num_groups),
        )
        scale_expand = self._reshape(ori_node.input[1], [-1, 1, 1])
        bias_expand = self._reshape(ori_node.input[2], [-1, 1, 1])
        mul = make_node(
            "Mul",
            inputs=[f"{ori_node.name}/Mul_input0", scale_expand.output[0]],
            outputs=[f"{ori_node.name}/Mul_output0"],
            name=f"{ori_node.name}/Mul",
        )
        add = make_node(
            "Add",
            inputs=[mul.output[0], bias_expand.output[0]],
            outputs=[ori_node.output[0]],
            name=f"{ori_node.name}/Add",
        )
        gn.output[0] = mul.input[0]
        self += [mul, add, ones, zeros]
        return gn

    def _reshape(self, node_input, target_shape):
        shape_cst = make_constant(
            f"{node_input}/Reshape/shape", np.array(target_shape, np.int64)
        )
        reshape = make_node(
            "Reshape",
            inputs=[node_input, shape_cst.output[0]],
            outputs=[f"{node_input}/Reshape_output0"],
            name=f"{node_input}/Reshape",
        )
        self += [shape_cst, reshape]
        return reshape
