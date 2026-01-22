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

from typing import List

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register(name="swap_concat", deps=["initializer_to_constant"])
class SwapConcat(Rewriter):
    """Swap concat op with Dequantize"""

    def __init__(self):
        pattern = GraphPattern()
        dq1 = SingleNodePattern("DequantizeLinear").with_attr("axis", 0)
        dq2 = SingleNodePattern("DequantizeLinear").with_attr("axis", 0)
        concat = SingleNodePattern("Concat")

        pattern.add_edge(dq1, concat)
        pattern.add_edge(dq2, concat)
        super().__init__(pattern)

    def make_concat(self, input_name_list: List[str], name: str, axis: int = 0):
        """concat two input"""
        concat = make_node(
            op_type="Concat",
            inputs=input_name_list[:],
            outputs=[f"{name}/Output"],
            name=name,
            axis=axis,
        )
        return concat

    def assign_nodes(self, nodes: List[NodeProto]):
        """Assign nodes by op_type, we find the order is not topo"""
        dq1, dq2, concat = None, None, None

        # assign concat
        for node in nodes:
            if node.op_type == "Concat":
                concat = node
        assert concat

        # assign dequantize by concat input
        dq_outputs = {
            node.output[0]: node for node in nodes if node.op_type == "DequantizeLinear"
        }
        for name in concat.input:
            if name in dq_outputs:
                if dq1 is None:
                    dq1 = dq_outputs[name]
                else:
                    dq2 = dq_outputs[name]

        assert dq1
        assert dq2

        return dq1, dq2, concat

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        if len(nodes) != 3:
            return
        dq1, dq2, concat = self.assign_nodes(nodes)

        # check continuous
        dq1_pos = [i for i, name in enumerate(concat.input) if name == dq1.output[0]][0]
        dq2_pos = [i for i, name in enumerate(concat.input) if name == dq2.output[0]][0]
        if dq1_pos + 1 != dq2_pos:
            return

        # only support weight concat
        if self.get_attribute(concat, "axis") != 0:
            return

        # concat weight on out channel(axis=0)
        x_node = self.make_concat(
            [dq1.input[0], dq2.input[0]], name=f"{dq1.input[0]}/MergedX"
        )
        dq1.input[0] = x_node.output[0]

        # concat scale
        scale_node = self.make_concat(
            [dq1.input[1], dq2.input[1]], name=f"{dq1.input[1]}/MergedScale"
        )
        dq1.input[1] = scale_node.output[0]

        # concat zero point
        zp_node = self.make_concat(
            [dq1.input[2], dq2.input[2]], name=f"{dq1.input[2]}/MergedZeroPoint"
        )
        dq1.input[2] = zp_node.output[0]

        # pop dq2
        concat.input.pop(dq2_pos)

        self += [x_node, scale_node, zp_node]
        self -= [dq2]
