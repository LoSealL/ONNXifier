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
from onnx.helper import make_tensor_type_proto, make_value_info

from ... import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(
    name="expand_qconv_channel", deps=["initializer_to_constant", "infer_shape"]
)
class ExpandQConvChannelRewriter(Rewriter):
    """Expand QConv's channel to 4"""

    def __init__(self):
        pattern = GraphPattern()
        qpattern = SingleNodePattern("QuantizeLinear")
        dqpattern = SingleNodePattern("DequantizeLinear")
        conv = SingleNodePattern("Conv")
        pattern.add_edge(qpattern, dqpattern)
        pattern.add_edge(dqpattern, conv)
        super().__init__(pattern=pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List):
        q_node, _, conv_node = nodes
        n, ic, h, w = graph.tensor_shape(conv_node.input[0])
        if ic != 3:
            return
        old_inputs = list(filter(lambda x: x.name == q_node.input[0], graph.input))
        if len(old_inputs) != 1:
            return
        old_input = old_inputs[0]
        # make sure that only one node connects with the old_input
        # TODO: support that multiple ops use old_input
        nodes = list(
            filter(lambda x: old_input.name in x["pb"].input, graph.nodes.values())
        )
        if len(nodes) != 1:
            return
        # make sure that only one node connects with the old_input
        # TODO: support that multiple ops use old_input
        nodes = list(
            filter(lambda x: old_input.name in x["pb"].input, graph.nodes.values())
        )
        if len(nodes) != 1:
            return

        # expand weight
        weight_dq_node = self.get_input_node_or_die(conv_node, 1)
        weight_node = self.get_input_node_or_die(weight_dq_node, 0)
        weight_value = self.get_value_or_die(weight_node)
        oc, _, kh, kw = weight_value.shape
        expand_weight_value = np.zeros([oc, 4, kh, kw], dtype=weight_value.dtype)
        expand_weight_value[:, :3, :, :] = weight_value
        # set zero value by zero point
        zero_point_value = self.get_value_or_die(
            self.get_input_node_or_die(weight_dq_node, 2)
        )
        expand_weight_value[:, 3, :, :] = np.reshape(zero_point_value, [oc, 1, 1])

        expand_weight_node = make_constant(
            weight_node.name + "/expand", expand_weight_value
        )

        # connect weight
        weight_dq_node.input[0] = expand_weight_node.output[0]
        # change input dim
        q_node.input[0] += "_expand"

        # replace graph input
        graph.input.append(
            make_value_info(q_node.input[0], make_tensor_type_proto(1, [n, 4, h, w]))
        )
        graph.input.remove(old_input)
        graph.inputs.pop(old_input.name)
        self -= weight_node
        self += expand_weight_node
