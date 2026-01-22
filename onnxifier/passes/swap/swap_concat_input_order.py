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

from collections.abc import Sequence
from typing import List

import numpy as np
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(name="swap_concat_input_order")
class SwapConcatInputOrderRewriter(Rewriter):
    """Swap the order of concat inputs.

    This rewriter is only acceptable if concat axis is 1.

    Before:

        concat(a, b, c)

    After:

        concat(c, b, a)  # [2, 1, 0]
    """

    def __init__(self):
        pattern = GraphPattern()
        concat = SingleNodePattern("Concat").with_attr("axis", 1)
        conv = SingleNodePattern("Conv")
        pattern.add_edge(concat, conv)
        super().__init__(pattern)

    # pylint: disable=arguments-differ
    def rewrite(
        self,
        graph: OnnxGraph,
        nodes: List[NodeProto],
        order: Sequence[int] | None = None,
    ):
        if not order:
            raise ValueError("swap order is not specified, skip rewriter")
        order = list(order)
        if len(order) != len(nodes[0].input):
            raise ValueError(
                f"swap order {order} must be same as concat input number "
                f"{len(nodes[0].input)}."
            )

        concat, conv = nodes
        if self.get_attribute(conv, "group", 1) != 1:
            raise ValueError("Conv group is not 1, skip rewriter")

        weight_data = self.get_value_or_die(conv.input[1])
        channels = [graph.static_tensor_shape(i)[1] for i in concat.input]
        new_inputs = np.asarray(concat.input)[order].tolist()
        concat.input[:] = new_inputs
        indices = np.cumsum(channels)[:-1]

        weight_sections = np.split(weight_data, indices, axis=1)
        new_weight = np.concatenate([weight_sections[i] for i in order], axis=1)
        weight_node = make_constant(f"{conv.name}/weight_shuffled", new_weight)
        conv.input[1] = weight_node.output[0]
        self += weight_node
