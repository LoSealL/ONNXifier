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
from onnx.onnx_pb import NodeProto

from . import IR_PASSES, OnnxGraph, Rewriter, SingleNodePattern, make_constant


@IR_PASSES.register("shapeof_to_constant")
class ShapeOfToConstantRewriter(Rewriter):
    """Calculate the result of ShapeOf node and replace it with a Constant node.

    If shape is dynamic, do nothing.
    """

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("ShapeOf"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        input_shape, _ = graph.tensor_info(node.input[0])
        if input_shape is None:
            return
        elif any(isinstance(i, str) or i <= 0 for i in input_shape):
            return  # dynamic shape

        self -= node
        if self.get_attribute(node, "output_type") == "i32":
            dtype = "int32"
        else:
            dtype = "int64"
        cst_node = make_constant(node.name, np.array(input_shape, dtype))
        cst_node.output[0] = node.output[0]
        self += cst_node
