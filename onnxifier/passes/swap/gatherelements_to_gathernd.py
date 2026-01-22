"""
Copyright (C) 2024 The ONNXIFIER Authors.

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
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(
    name="gatherelements_to_gathernd", deps=["initializer_to_constant", "infer_shape"]
)
class GatherElementsToGatherNDRewrite(Rewriter):
    """convert GatherElements op (and tile op) to GatherND op"""

    def __init__(self):
        pattern = GraphPattern()
        tile = SingleNodePattern("Tile")
        gatherElements = SingleNodePattern("GatherElements").with_attr("axis", 1)

        pattern.add_edge(tile, gatherElements)
        super().__init__(pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        tile, gatherElements = nodes

        # check
        input_shape = graph.tensor_shape(gatherElements.input[0])
        index_shape = graph.static_tensor_shape(tile.input[0])
        if len(index_shape) != 3:
            return
        # this pass only support last dim repeat
        repeats = self.get_value(tile.input[1])
        if np.any(np.array([1, 1, input_shape[-1]]) != np.array(repeats)):
            return

        # zero constant
        zero_constant = make_constant(
            f"{tile.name}/zero_constant", np.zeros(index_shape, dtype="int64")
        )
        # new concat
        concat = make_node(
            "Concat",
            inputs=[zero_constant.output[0], tile.input[0]],
            outputs=[f"{tile.name}/Concat_Output"],
            name=f"{tile.name}/Concat",
            axis=2,
        )
        gather = make_node(
            "GatherND",
            inputs=[gatherElements.input[0], concat.output[0]],
            outputs=[gatherElements.output[0]],
            name=f"{gatherElements.name}/GatherND",
            batch_dims=0,
        )

        self += [zero_constant, concat, gather]
        self -= [tile, gatherElements]
