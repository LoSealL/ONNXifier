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

# pylint: disable=arguments-differ

from typing import List

from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register(name="resize_to_nearest_neighbor")
class ResizeToNearestNeighborRewriter(Rewriter):
    """Simplify any resize with integer scales to nearest neighbor interpolate"""

    def __init__(self):
        super().__init__(SingleNodePattern("Resize"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node_pb = nodes[0]
        mode = self.get_attribute(node_pb, "mode")
        if mode != "nearest":
            self.set_attribute(node_pb, "mode", "nearest")
        self._simplify_coordinate_transformation_mode(node_pb)

    def _simplify_coordinate_transformation_mode(self, node_pb):
        self.set_attribute(node_pb, "coordinate_transformation_mode", "asymmetric")
