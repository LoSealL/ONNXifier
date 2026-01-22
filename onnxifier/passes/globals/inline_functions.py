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

# pylint: disable=arguments-differ

import json
from copy import deepcopy
from typing import List
from uuid import uuid4

from onnx.inliner import inline_local_functions
from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("inline_local_functions")
def inline_local_functions_pass(graph: OnnxGraph) -> OnnxGraph:
    """Inline all local functions in the graph."""
    return OnnxGraph(inline_local_functions(graph.model), base_dir=graph.external_base)


@PASSES.register("inline_functions")
class InlineFunctionsRewriter(Rewriter):
    """This pass inlines all functions into the main graph."""

    def __init__(self):
        super().__init__(SingleNodePattern().with_domain("*"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], force: bool = True):
        node = nodes[0]
        if node.op_type not in graph.functions:
            raise RuntimeError(f"function {node.op_type} not found in the graph")

        try:
            node_names = json.loads(node.doc_string)
            assert isinstance(node_names, list)
            assert all(map(lambda x: isinstance(x, str), node_names))
        except Exception:
            node_names = []
        if len(node_names) == 0 and not force:
            return

        tag = f"{node.name}/{node.op_type}"

        # step 1: find the input and output port that matches the custom node
        func = graph.functions[node.op_type]
        func_nodes = [deepcopy(node) for node in func.node]
        assert len(func_nodes) == len(node_names) or len(node_names) == 0

        for n in func_nodes:
            for j, node_input in enumerate(n.input):
                if node_input in func.input:
                    n.input[j] = node.input[list(func.input).index(node_input)]
                else:
                    n.input[j] = f"{tag}/{n.input[j]}"
        for n in func_nodes:
            for j, node_output in enumerate(n.output):
                if node_output in func.output:
                    n.output[j] = node.output[list(func.output).index(node_output)]
                else:
                    n.output[j] = f"{tag}/{n.output[j]}"
        # step 2: make a copy of each function node appended with identity names
        for i, n in enumerate(func_nodes):
            if not n.name:
                n.name = f"{uuid4().hex}"
            n.name = node_names[i] if node_names else f"{tag}/{n.name}"

        # step 3: assign attributes
        for attr in node.attribute:
            value = self.get_attribute(node, attr.name)
            for n in func_nodes:
                for attr2 in n.attribute:
                    if attr2.ref_attr_name == attr.name:
                        self.set_attribute(n, attr2.name, value)

        self += func_nodes
        self -= node
