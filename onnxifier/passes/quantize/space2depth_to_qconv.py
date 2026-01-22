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
from typing import List

import numpy as np
from onnx.helper import make_node, tensor_dtype_to_np_dtype
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(name="space2depth_to_qconv", deps=["initializer_to_constant"])
class SpaceToDepthToQConvRewriter(Rewriter):
    """Convert SpaceToDepth - QDQ to quantized Conv."""

    def __init__(self):
        qdq_pattern = GraphPattern()
        s2d = SingleNodePattern("SpaceToDepth")
        qpattern = SingleNodePattern("QuantizeLinear")
        dqpattern = SingleNodePattern("DequantizeLinear")
        qdq_pattern.add_edge(s2d, qpattern)
        qdq_pattern.add_edge(qpattern, dqpattern)
        super().__init__(pattern=qdq_pattern)

    def _copy_qdq(self, node: NodeProto) -> NodeProto:
        axis = self.get_attribute(node, "axis") or 0
        scale_node = self.get_input_node_or_die(node, 1)
        scale_value = self.get_value_or_die(scale_node)
        new_scale = make_constant(node.name + "/scale/copy", scale_value)
        input_names = [node.input[0] + "/copy", new_scale.output[0]]
        if len(node.input) == 3:
            zero_node = self.get_input_node_or_die(node, 2)
            zero_value = self.get_value_or_die(zero_node)
            new_zero = make_constant(node.name + "/zero/copy", zero_value)
            self += new_zero
            input_names += new_zero.output[:]
        new_node = make_node(
            node.op_type,
            input_names,
            [o + "/copy" for o in node.output],
            node.name + "/s2d_to_qconv/copy",
            axis=axis,
        )
        self += [new_scale]
        return new_node

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        s2d, qnode, dqnode = nodes
        blocksize = self.get_attribute(s2d, "blocksize")
        assert isinstance(blocksize, int)
        _, ic, _, _ = graph.tensor_shape(s2d.input[0])
        _, oc, _, _ = graph.tensor_shape(s2d.output[0])
        assert isinstance(ic, int) and isinstance(oc, int)
        etype = graph.tensor_type(dqnode.input[0])
        dtype = tensor_dtype_to_np_dtype(etype)
        assert ic * blocksize**2 == oc, "invalid space2depth parameters"

        qnode_ahead = self._copy_qdq(qnode)
        dqnode_ahead = self._copy_qdq(dqnode)
        qnode_ahead.input[0] = s2d.input[0]
        dqnode_ahead.input[0] = qnode_ahead.output[0]

        kernel = (
            # fmt: off
            np.tile(
                np.eye(blocksize**2)
                .reshape([-1, blocksize, blocksize])
                .astype(dtype),
                [ic, 1, 1, 1],
            )
            .transpose([1, 0, 2, 3])
            .reshape([oc, 1, blocksize, blocksize])
            # fmt: on
        )
        # expand to [oc, ic, blocksize, blocksize]
        kernel = np.concatenate(
            [kernel, np.tile(np.zeros_like(kernel), [1, ic - 1, 1, 1])], axis=1
        )
        for i in range(1, ic):
            kernel[i::ic][:, i] = kernel[i::ic][:, 0]
            kernel[i::ic][:, 0] = 0  # swap
        kernel_max = np.iinfo(dtype).max
        conv_weight = make_constant(
            name=f"{s2d.name}/weight", value=kernel * kernel_max
        )

        # kernel dequantize
        kernel_scale = make_constant(
            conv_weight.name + "/scale", np.ones([oc], np.float32) / kernel_max
        )
        kernel_zp = make_constant(conv_weight.name + "/zero", np.zeros([oc], dtype))
        kernel_dq = make_node(
            "DequantizeLinear",
            [conv_weight.output[0], kernel_scale.output[0], kernel_zp.output[0]],
            [conv_weight.output[0] + "/dq"],
            conv_weight.name + "/dequantize",
            axis=0,
        )

        conv = make_node(
            op_type="Conv",
            inputs=dqnode_ahead.output[:] + [kernel_dq.output[0]],
            outputs=s2d.output,
            name=f"{s2d.name}/conv",
            kernel_shape=[blocksize, blocksize],
            strides=[blocksize, blocksize],
        )
        self -= s2d
        self += [
            qnode_ahead,
            dqnode_ahead,
            kernel_dq,
            kernel_scale,
            kernel_zp,
            conv_weight,
            conv,
        ]
