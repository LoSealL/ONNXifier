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

from typing import List, Protocol, Type

from onnx import NodeProto

from .graph import OnnxGraph
from .passes.rewriter import Rewriter


class RewriterInterface(Protocol):
    """Interface helper for rewriting ONNX graphs."""

    __NAME__: str
    """Specify the name of the rewriter pass. Defaults to class or function name."""

    __DEPS__: List[str | Type[Rewriter]]
    """Specify the dependencies of the rewriter pass."""

    __PATCHES__: List[str | Type[Rewriter]]
    """Specify the finalizers of the rewriter pass."""

    def __call__(self, graph: OnnxGraph, *args, **kwargs) -> OnnxGraph:
        """Rewriter is a callable that takes :class:`OnnxGraph` and returns a
        modified graph.
        """
        return graph

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        """Implement how to rewrite matched nodes in the graph

        Args:
            graph (OnnxGraph): an onnx graph
            nodes (List[NodeProto]): a list of matched nodes
        """

    @property
    def num_rewrites(self) -> int:
        """This property records how many patterns have been matched and rewritten"""
        return 0
