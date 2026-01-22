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
from onnx import NodeProto, numpy_helper
from onnx.helper import make_node

from ... import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(name="constant_to_constantofshape")
class ConstantToConstantOfShapeRewriter(Rewriter):
    """Rewrite Constant to ConstantOfShape node.

    This pass reduces ONNX model size by converting large constant tensors
    to ConstantOfShape nodes when the number of elements exceeds threshold.
    """

    def __init__(self, threshold: int = 16):
        """Initialize the rewriter.

        Args:
            threshold: Only convert constants with more than this many elements.
        """
        super().__init__(pattern=SingleNodePattern("Constant"))
        self.threshold = threshold

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]

        # Get the constant value
        constant_value = self.get_value(node)
        if constant_value is None:
            return

        # Only convert if the number of elements exceeds threshold
        if constant_value.size <= self.threshold:
            return

        # Take the first element as the scalar value to reduce model size
        scalar_value = constant_value.flat[0]

        # Create shape constant for the shape input
        shape_array = np.array(constant_value.shape, dtype=np.int64)
        shape_node = make_constant(node.name + "/shape", shape_array)

        # Create ConstantOfShape node
        # The value attribute should be a tensor with single element
        value_tensor = numpy_helper.from_array(
            np.array([scalar_value], dtype=constant_value.dtype)
        )
        constantofshape_node = make_node(
            op_type="ConstantOfShape",
            inputs=[shape_node.output[0]],
            outputs=node.output,
            name=node.name + "/constantofshape",
            value=value_tensor,
        )

        # Replace the original constant node
        self -= node
        self += shape_node
        self += constantofshape_node
