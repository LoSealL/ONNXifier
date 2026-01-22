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

from onnx import NodeProto, TensorProto
from onnx.helper import make_node

from ...graph import OnnxGraph
from .. import PASSES, Rewriter
from ..pattern import SingleNodePattern


@PASSES.register(name="castlike_to_cast", deps=["infer_shape"])
class CastLikeToCastRewriter(Rewriter):
    """Convert CastLike to Cast, if data type is known to be static."""

    def __init__(self):
        super().__init__(SingleNodePattern("CastLike"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        dtype = graph.tensor_type(node.input[1])
        if dtype == TensorProto.UNDEFINED:
            return

        cast = make_node(
            "Cast",
            inputs=[node.input[0]],
            outputs=node.output,
            to=dtype,
            name=node.name,
        )

        self += cast
        self -= node
