"""
Copyright (C) 2025 The ONNX2ONNX Authors.

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

from itertools import chain
from pathlib import Path
from typing import List

import onnx

from ... import OnnxGraph, logger
from .. import PASSES
from ..pattern import OutputNodePattern
from ..rewriter import Rewriter
from ..utils import canonical_node_name


@PASSES.register("split_non_compute_outputs", deps=["infer_shape"])
class SplitNonComputeOutputsRewriter(Rewriter):
    """Find non compute intensive nodes adjacent to the outputs and split them out.

    Example::

        Before:
            Conv -> Sigmoid -> Concat -<
                     Conv   __/

        After:
            1. Conv -<
               Conv -<
            2. Sigmoid -<

    """

    _compute_op = ("Conv", "ConvTranspose", "DeformConv", "Gemm", "MatMul")
    _fusing_act_op = ("Relu", "LeakyRelu", "PRelu")

    def __init__(self):
        self.model_cache = (
            Path("~").expanduser() / f".cache/onnx2onnx/{self.__class__.__name__}"
        )
        self.model_cache.mkdir(exist_ok=True, parents=True)
        super().__init__(OutputNodePattern())

    def _can_be_removed(self, graph: OnnxGraph, node, nodes_removed):
        if node.op_type in chain(self._compute_op, self._fusing_act_op):
            return False
        # all downstream nodes have been removed
        return all(i in nodes_removed for i in graph.onnx_successors(node))

    def rewrite(self, graph: OnnxGraph, nodes: List[onnx.NodeProto], *args, **kwargs):
        node = nodes[0]
        if node.op_type in self._compute_op:
            logger.debug(f"Skip node {node.name} since it's {node.op_type}")
            return
        nodes_to_remove: List[onnx.NodeProto] = []
        nodes_to_visit = [node]
        while nodes_to_visit:
            node_to_remove = nodes_to_visit.pop(0)
            if self._can_be_removed(graph, node_to_remove, nodes_to_remove):
                nodes_to_remove.append(node_to_remove)
                nodes_to_visit.extend(graph.onnx_predecessors(node_to_remove))
        # save the deleted nodes
        model_save = self.model_cache / f"{canonical_node_name(node.name)}.onnx"
        onnx.save_model(graph.onnx_subgraph(nodes_to_remove).model, model_save)
        logger.debug(f"Split out {len(nodes_to_remove)} nodes to {model_save}")
        self -= nodes_to_remove
        # add new output ports
        for leaf_node in filter(
            lambda i: i not in nodes_to_remove,
            chain(*[graph.onnx_predecessors(node) for node in nodes_to_remove]),
        ):
            for output_name in leaf_node.output:
                logger.debug(f"Add output to {leaf_node.name}:{output_name}")
                graph.set_output(leaf_node, output_name)
