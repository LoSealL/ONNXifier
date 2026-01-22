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

import networkx as nx
import numpy as np
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph, logger
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("random_initialize_weight", deps=["initializer_to_constant"])
class RandomInitializeWeightRewriter(Rewriter):
    """Randomly initialize Conv and ConvTranspose weights."""

    def __init__(self):
        super().__init__(SingleNodePattern("Conv") | SingleNodePattern("ConvTranspose"))

    def _get_constant(self, graph: OnnxGraph, node):
        pred = self.get_input_node(node, 1)
        assert pred is not None
        for n in nx.ancestors(graph, pred.name):
            if graph.nodes[n]["pb"].op_type == "Constant":
                yield graph.nodes[n]["pb"]

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        for c in self._get_constant(graph, node):
            value = self.get_value(c)
            assert value is not None
            if value.ndim < 4:
                continue
            value = np.random.uniform(
                low=value.min(), high=value.max(), size=value.shape
            ).astype(value.dtype)
            self.set_attribute(c, "value", value)
            logger.debug(f"initialize {c.name} on {node.name}")
