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

from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("mute_maxpool_indices")
class MuteMaxPoolIndicesRewriter(Rewriter):
    """Disable the 2nd output of MaxPool, a.k.a indices. If there is not any
    downstream node uses the indices.

    Some devices like NPU may not recognize the indices output of MaxPool.
    """

    def __init__(self):
        super().__init__(SingleNodePattern("MaxPool"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        if len(node.output) == 1:
            return

        if not self.get_output_node(node, 1):
            # no downstream node uses the indices
            node.output.pop(1)
