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


@PASSES.register("quick_sparse", deps=["initializer_to_constant"])
class QuickSparseRewriter(Rewriter):
    """Randomly sparse the weight of Conv and ConvTranspose layers.

    Note:
        This is only for experimental purpose. It is not a real pruning!
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Conv") | SingleNodePattern("ConvTranspose"))

    def _find_node_ancestors(self, graph: OnnxGraph, root: NodeProto, op_type: str):
        if root.op_type == op_type:
            yield root
        for n in nx.ancestors(graph, root.name):
            if graph.nodes[n]["pb"].op_type == op_type:
                yield graph.nodes[n]["pb"]

    def rewrite(
        self,
        graph: OnnxGraph,
        nodes: List[NodeProto],
        structured: bool = False,
        ratio: float = 0.5,
    ):
        if ratio < 0.1:
            logger.warning(f"sparsity ratio < 10 is meaningless, got {ratio:%}.")
            return
        node = nodes[0]
        weight = self.get_input_node(node, 1)
        if weight is None:
            return
        # group = self.get_attribute(node, "group") or 1

        for c in self._find_node_ancestors(graph, weight, "Constant"):
            value = self.get_value(c)
            assert value is not None
            value = value.copy()
            if value.ndim < 4:
                continue
            out_channel, in_channel = value.shape[:2]
            if node.op_type == "ConvTranspose":
                in_channel, out_channel = value.shape[:2]
            if structured:
                mask = np.random.uniform(0, 1, size=in_channel) >= ratio
                shape = list(value.shape)
                shape[1] = 1
                mask = np.tile(mask.reshape([1, -1, 1, 1]), shape)
                if node.op_type == "ConvTranspose":
                    shape[0] = 1
                    mask = np.tile(mask.reshape([-1, 1, 1, 1]), shape)
                assert mask.shape == value.shape
            else:
                mask = np.random.uniform(0, 1, size=value.shape) >= ratio
            if value.dtype in (np.float32, np.float16):
                value *= mask.astype(value.dtype)
            else:
                np.iinfo(value.dtype)
                dq = next(self._find_node_ancestors(graph, weight, "DequantizeLinear"))
                zero_point = self.get_value(dq.input[2])
                if zero_point is None:
                    zero_point = np.zeros([out_channel], dtype=value.dtype)
                value *= mask.astype(value.dtype)
                zero_point = zero_point.reshape([out_channel, 1, 1, 1])
                if node.op_type == "ConvTranspose":
                    zero_point = zero_point.reshape([1, out_channel, 1, 1])
                value += zero_point * np.bitwise_not(mask).astype(value.dtype)
            self.set_attribute(c, "value", value)
