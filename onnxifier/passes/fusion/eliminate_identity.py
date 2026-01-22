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

from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import L2
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@L2.register(name="eliminate_identity", deps=["replicate_identity_constant"])
class EliminateIdentityRewriter(Rewriter):
    """Remove Identity nodes."""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Identity"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        identity_node = nodes[0]
        if graph.nodes[identity_node.name]["has_output"]:
            prev_node = graph.onnx_predecessors(identity_node)[0]
            for i, output_name in enumerate(prev_node.output):
                if output_name == identity_node.input[0]:
                    prev_node.output[i] = identity_node.output[0]
            # Since identity's input is replaced by its output, every sibling node's
            # input should also be replaced.
            sibling_nodes = graph.onnx_siblings(identity_node)
            for node in sibling_nodes:
                for i, input_name in enumerate(node.input):
                    if input_name == identity_node.input[0]:
                        node.input[i] = identity_node.output[0]
            # replace output node
            graph.set_output(prev_node, identity_node.output[0])
            self -= identity_node
            return
        for node in graph.onnx_successors(identity_node):
            for i, input_name in enumerate(node.input):
                if input_name == identity_node.output[0]:
                    node.input[i] = identity_node.input[0]
        self -= identity_node
