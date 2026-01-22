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

import networkx as nx
from onnx import NodeProto

from ... import OnnxGraph
from ...evaluator import Evaluator
from .. import L1
from ..pattern import ConstantGraphPattern
from ..rewriter import Rewriter
from ..utils import make_constant


@L1.register(
    name="fold_constant",
    deps=["initializer_to_constant", "infer_shape", "shape_to_constant"],
)
class FoldConstantPass(Rewriter):
    """Fold constants to a single node."""

    def __init__(self):
        super().__init__(pattern=ConstantGraphPattern())

    def _is_constant_or_qdq(self, node):
        return node.op_type in {"Constant", "DequantizeLinear", "QuantizeLinear"}

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        # skip if all nodes are Constant
        if all(self._is_constant_or_qdq(node) for node in nodes):
            return
        subonnx = graph.onnx_subgraph(nodes)
        out_nodes = {}
        constants = []
        for node_name in nx.topological_sort(subonnx):
            node_pb = subonnx.nodes[node_name]["pb"]
            if subonnx.nodes[node_name]["has_output"]:
                for out_name in node_pb.output:
                    if out_name in subonnx.outputs:
                        out_nodes[out_name] = node_pb
        # filter out the outputs that are exactly constant's output
        outputs_eval = set(subonnx.outputs)
        for node in list(filter(lambda n: n.op_type == "Constant", nodes)):
            if set(node.output).issubset(outputs_eval):
                nodes.remove(node)
                outputs_eval.difference_update(node.output)
        evaluator = Evaluator(subonnx.model)
        outputs = evaluator(list(outputs_eval), {})
        for output_name, out_value in zip(outputs_eval, outputs):
            constants.append(make_constant(output_name, out_value))
            constants[-1].output[0] = output_name  # keep the edge name
        self -= nodes
        self += constants
