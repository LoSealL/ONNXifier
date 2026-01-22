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

from abc import abstractmethod
from copy import deepcopy
from typing import List, Optional

from onnx.onnx_pb import NodeProto

from ...passes import (
    IR_PASSES,
    OnnxGraph,
    Pattern,
    Registry,
    Rewriter,
    RewriterRepeat,
    SingleNodePattern,
)


class BaseNodeConversion(Rewriter):
    """Abstract class for a single IR node conversion."""

    def __init__(self, pattern: Optional[Pattern] = None, repeat=RewriterRepeat.ONCE):
        if pattern is None:
            pattern = SingleNodePattern(self.__class__.__name__).with_attr("version")
        super().__init__(pattern=pattern, repeat=repeat)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        new_node = self.replace(graph, deepcopy(node))
        self += new_node

    @abstractmethod
    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        """Replace the ori_node to ai.onnx operator"""
        raise NotImplementedError("replace method is not implemented")


OP_CONVERT = Registry("OP_CONVERT", parent=IR_PASSES)
