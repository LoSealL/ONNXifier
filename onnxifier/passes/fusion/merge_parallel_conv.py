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

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register(name="merge_parallel_conv", deps=["initializer_to_constant"])
class MergeParallelConvRewrite(Rewriter):
    """Merge parallel conv ops into one conv op"""

    def __init__(self):
        pattern = GraphPattern()
        conv1 = SingleNodePattern("Conv")
        conv2 = SingleNodePattern("Conv")
        concat = SingleNodePattern("Concat")

        pattern.add_edge(conv1, concat)
        pattern.add_edge(conv2, concat)
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
        conv1, conv2, concat = None, None, None

        # assign concat
        for node in nodes:
            if node.op_type == "Concat":
                concat = node
        assert concat

        # assign conv by concat input
        conv_outputs = {
            node.output[0]: node for node in nodes if node.op_type == "Conv"
        }
        for name in concat.input:
            if name in conv_outputs:
                if conv1 is None:
                    conv1 = conv_outputs[name]
                else:
                    conv2 = conv_outputs[name]

        assert conv1
        assert conv2

        return conv1, conv2, concat

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        if len(nodes) != 3:
            return
        conv1, conv2, concat = self.assign_nodes(nodes)

        # TODO: check conv attrs
        # check continuous
        conv1_pos = [
            i for i, name in enumerate(concat.input) if name == conv1.output[0]
        ][0]
        conv2_pos = [
            i for i, name in enumerate(concat.input) if name == conv2.output[0]
        ][0]
        if conv1_pos + 1 != conv2_pos:
            return

        # check same input
        if conv1.input[0] != conv2.input[0]:
            return

        # both have bias or not
        if len(conv1.input) != len(conv2.input):
            return

        # only support channel concat
        if self.get_attribute(concat, "axis") != 1:
            return

        # concat weight on out channel(axis=0)
        w_node = self.make_concat(
            [conv1.input[1], conv2.input[1]], name=f"{conv1.input[1]}/MergedWeight"
        )
        conv1.input[1] = w_node.output[0]
        self += w_node

        # concat bias
        if len(conv1.input) == 3:
            b_node = self.make_concat(
                [conv1.input[2], conv2.input[2]], name=f"{conv1.input[2]}/MergedBias"
            )
            conv1.input[2] = b_node.output[0]
            self += b_node

        # pop conv2
        concat.input.pop(conv2_pos)

        self -= [conv2]
