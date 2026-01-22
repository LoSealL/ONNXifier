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

from typing import List

from onnx.onnx_pb import NodeProto

from ... import OnnxGraph, logger
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("eliminate_reshape", deps=["infer_shape"])
class EliminateInputOutputReshapeRewriter(Rewriter):
    """Eliminate useless reshape operations."""

    def __init__(self):
        super().__init__(SingleNodePattern("Reshape"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        self._rewrite_input_reshape(graph, node)
        self._rewrite_output_reshape(graph, node)

    def _rewrite_input_reshape(self, graph: OnnxGraph, node: NodeProto):
        """remove the reshape if it is the input of the graph

        Before:
            input(shape1) -> reshape(shape2) -> conv

        After:
            input(shape2) -> conv
        """
        if node.input[0] in graph.inputs:
            input_name = node.input[0]
            graph.remove_input(input_name)
            for n in graph.onnx_successors(node):
                graph.set_input(n, node.output[0])
                input_pos = [i for i in n.input].index(node.output[0])
                logger.debug(f"set input {input_pos} of {n.name} to {input_name}")
                graph.rename_input(node.output[0], input_name)
                n.input[input_pos] = input_name
            self -= node

    def _rewrite_output_reshape(self, graph: OnnxGraph, node: NodeProto):
        """remove the reshape if it is the output of the graph

        Before:
            conv(shape1) -> reshape(shape2) -> output(shape2)

        After:
            conv -> output(shape1)
        """
        if node.output[0] in graph.outputs:
            output_name = node.output[0]
            graph.remove_output(output_name)
            for n in graph.onnx_predecessors(node):
                graph.set_output(n, node.input[0])
                output_pos = [i for i in n.output].index(node.input[0])
                logger.debug(f"set output {output_pos} of {n.name} to {output_name}")
                graph.rename_output(node.input[0], output_name)
                n.output[output_pos] = output_name
            self -= node
