"""
Copyright (C) 2026 The ONNXIFIER Authors.

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

import numpy as np
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(name="shape_to_constant")
class ShapeToConstantPass(Rewriter):
    """Convert static Shape op output to Constant."""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Shape"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        try:
            shape = graph.static_tensor_shape(node.input[0])
        except ValueError:
            # shape is not constant
            return
        start = self.get_attribute(node, "start", 0)
        end = self.get_attribute(node, "end", len(shape))
        shape = np.array(shape, "int64")[start:end]
        # replace Shape with Constant
        shape_const = make_constant(node.name + "/Shape", shape)
        shape_const.output[0] = node.output[0]
        self -= node
        self += shape_const
