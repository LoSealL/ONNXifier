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

from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(name="replicate_identity_constant")
class ReplicateIdentityConstantRewriter(Rewriter):
    """Replicate constant node that identifies to multiple destinations.

    Before:

        constant -> identity -> fanout1
                           | -> fanout2
                           | -> ...

    After:

        constant_copy0 -> fanout1
        constant_copy1 -> fanout2
        constant_copy2 -> ...
    """

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Identity"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        identity_node = nodes[0]
        constant_node = self.get_input_node(identity_node, 0)
        if constant_node is None:
            # get value from initializer
            value = self.get_value(identity_node.input[0])
            fanout = 0
        elif constant_node.op_type == "Constant":
            value = self.get_value(constant_node)
            fanout = len(graph.onnx_successors(constant_node))
        else:
            return
        if value is None:
            return  # not a constant

        for i, succ in enumerate(graph.onnx_successors(identity_node)):
            pos = list(succ.input).index(identity_node.output[0])
            # make a copy of constant
            constant_copy_node = make_constant(
                f"{identity_node.name}/constant_copy{i}",
                value,
            )
            succ.input[pos] = constant_copy_node.output[0]
            self += constant_copy_node

        self -= identity_node
        if fanout == 1 and isinstance(constant_node, NodeProto):
            self -= constant_node
