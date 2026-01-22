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

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register("replace_squeeze_to_reshape")
class ReplaceSqueezeToReshapeRewriter(Rewriter):
    """Replace squeeze to reshape."""

    def __init__(self):
        super().__init__(SingleNodePattern("Squeeze"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        output_shape = graph.static_tensor_shape(nodes[0].output[0])
        shape = make_constant(
            f"{nodes[0].name}/shape", np.array(output_shape, dtype=np.int64)
        )
        nodes[0].input[1] = shape.output[0]
        reshape = make_node(
            "Reshape",
            inputs=nodes[0].input,
            outputs=nodes[0].output,
            name=nodes[0].name,
        )
        self += [shape, reshape]
        self -= nodes[0]
