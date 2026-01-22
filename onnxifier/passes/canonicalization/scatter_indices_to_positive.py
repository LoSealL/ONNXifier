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

import numpy as np
import onnx

from ... import OnnxGraph, logger
from .. import L1
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@L1.register(name="scatter_nd_indices_to_positive", deps=["infer_shape"])
class ScatterNDIndicesToPositiveRewriter(Rewriter):
    r"""This pass converts potential negative indices in ScatterND op to
    positive indices.

    Before:

        ScatterND(data, indices, updates)

    After:

        ScatterND(data, indices_pos, updates)
    """

    def __init__(self):
        super().__init__(SingleNodePattern("ScatterND"))

    def rewrite(self, graph: OnnxGraph, nodes: List[onnx.NodeProto]):
        node = nodes[0]
        indices = self.get_value_or_die(node.input[1]).copy()
        data_shape = graph.static_tensor_shape(node.input[0])
        if np.all(indices >= 0):  # no negative indices
            return

        logger.debug(f"Found negative indices in ScatterND {node.name}")
        logger.debug(f"Data shape: {data_shape}")

        for i, d in enumerate(data_shape):
            indices[..., i] = np.where(
                indices[..., i] < 0, indices[..., i] + d, indices[..., i]
            )
        pos_indices = make_constant(f"{node.name}/indices_pos", indices)
        node.input[1] = pos_indices.output[0]
        self += pos_indices
