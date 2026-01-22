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
from onnx import NodeProto

from ... import OnnxGraph
from .. import L1
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@L1.register(name="constantofshape_to_constant")
class ConstantOfShapeRewriter(Rewriter):
    """Rewrite ConstantOfShape to a constant node."""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("ConstantOfShape"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        shape = self.get_value_or_die(node.input[0])
        value = self.get_attribute(node, "value")
        if value is None:  # default value is 0.0
            value = np.array([0], dtype=np.float32)
        assert isinstance(value, np.ndarray)
        assert len(value) == 1
        value = value[0]
        const_node = make_constant(
            node.name + "/const", np.zeros(shape, dtype=value.dtype) + value
        )
        const_node.output[0] = node.output[0]
        self -= node
        self += const_node
