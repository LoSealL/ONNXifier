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
from typing import Dict, List, Tuple, Union

import numpy as np
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph, logger
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register(name="inspect_sparsity_ratio", deps=["initializer_to_constant"])
class InspectSparsityRatio(Rewriter):
    """Inspect Sparsity Ratio"""

    _global_sparsity: Dict[int, Dict[str, Tuple[Tuple[int, ...], float]]] = {}
    """Global sparsity mapping, graph id to layer->(shape, sparsity ratio)"""

    def __init__(self):
        node = SingleNodePattern("Conv") | SingleNodePattern("ConvTranspose")
        super().__init__(node)
        self.register_post_hook(self._print_global_sparsity)

    def _print_global_sparsity(self, graph: OnnxGraph) -> OnnxGraph:
        sparse_map = self._global_sparsity.get(id(graph))
        if sparse_map:
            total_size: float = 0
            sparse_size: float = 0
            for shape, ratio in sparse_map.values():
                total_size += np.prod(shape, dtype="float")
                sparse_size += np.prod(shape, dtype="float") * ratio
            if total_size > (1 << 30):
                unit, unit_text = 1e9, "GiB"
            elif total_size > (1 << 20):
                unit, unit_text = 1e6, "MiB"
            elif total_size > (1 << 10):
                unit, unit_text = 1e3, "KiB"
            else:
                unit, unit_text = 1, "B"
            logger.info(f"Total size: {total_size / unit:.2f} {unit_text}")
            logger.info(f"Sparse size: {sparse_size / unit:.2f} {unit_text}")
            logger.info(f"Sparsity ratio: {sparse_size / total_size:.2%}")
        return graph

    def get_sparse_ratio(
        self,
        weight: NodeProto,
        zero_point: Union[int, np.ndarray] = 0,
        op_type: str = "Conv",
    ) -> Tuple[Tuple[int, ...], float]:
        """Inspect sparse ratio on specific location"""
        weight_value = self.get_value(weight)
        if weight_value is None:
            return (), -1

        # expand zero_point's shape
        if isinstance(zero_point, np.ndarray):
            new_shape = [1] * len(weight_value.shape)
            if op_type == "ConvTranspose":
                new_shape[1] = zero_point.size
            else:
                new_shape[0] = zero_point.size
            zero_point = zero_point.reshape(new_shape)
        sparse_ratio = np.count_nonzero(weight_value == zero_point) / weight_value.size
        return weight_value.shape, sparse_ratio

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        if id(graph) not in self._global_sparsity:
            self._global_sparsity[id(graph)] = {}
        sparse_map = self._global_sparsity[id(graph)]

        node = nodes[0]
        weight = self.get_input_node(node, 1)
        if weight is None:
            return

        sparse_ratio = -1.0
        if weight.op_type == "Constant":
            zero_point = 0
        elif weight.op_type == "DequantizeLinear":
            dequantize = weight
            # use DequantizeLinear's x as weight
            weight = self.get_input_node(dequantize, 0)
            if weight is None:
                return
            zero_point = self.get_value(dequantize.input[2])
            if zero_point is None:
                zero_point = 0
        else:
            return
        shape, sparse_ratio = self.get_sparse_ratio(weight, zero_point, node.op_type)

        if sparse_ratio == -1:
            return
        logger.info(f"{node.name}: {sparse_ratio:.2%}")
        sparse_map[node.name] = (shape, sparse_ratio)
