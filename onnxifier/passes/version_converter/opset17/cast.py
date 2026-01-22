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

from .... import OnnxGraph
from ...pattern import SingleNodePattern
from ...rewriter import Rewriter
from . import OP_CONVERTER


@OP_CONVERTER.register("Cast")
class Cast(Rewriter):
    """Remove "saturate" attribute from Cast op."""

    def __init__(self):
        super().__init__(SingleNodePattern("Cast"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        self.remove_attribute(node, "saturate")


@OP_CONVERTER.register("CastLike")
class CastLike(Rewriter):
    """Remove "saturate" attribute from CastLike op."""

    def __init__(self):
        super().__init__(SingleNodePattern("CastLike"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        self.remove_attribute(node, "saturate")
