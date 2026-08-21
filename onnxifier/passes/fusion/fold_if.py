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

from copy import deepcopy

import numpy as np
from onnx import AttributeProto, NodeProto
from onnx.onnx_pb import GraphProto

from ... import OnnxGraph
from .. import L2
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter, RewriterRepeat


@L2.register(name="fold_if", deps=["infer_shape"])
class FoldIfPass(Rewriter):
    """Inline the taken branch of an ``If`` node whose condition is constant."""

    def __init__(self):
        super().__init__(
            pattern=SingleNodePattern("If"), repeat=RewriterRepeat.INFINITE
        )

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto]):
        if_node = nodes[0]
        cond_name = if_node.input[0]
        cond_val = self.get_value(cond_name)
        if cond_val is None:
            return  # condition is not statically known

        take_then = bool(np.asarray(cond_val).reshape(-1)[0])

        then_attr = next(a for a in if_node.attribute if a.name == "then_branch")
        else_attr = next(a for a in if_node.attribute if a.name == "else_branch")
        branch = then_attr.g if take_then else else_attr.g

        self._inline(graph, if_node, branch)
        self -= if_node

    def _inline(self, graph: OnnxGraph, if_node: NodeProto, branch: GraphProto) -> None:
        """Copy ``branch`` into ``graph``, replacing the ``If`` node.

        The branch's declared inputs are mapped to the ``If`` node's inputs
        (only ``cond`` for ONNX ``If``) and its outputs to the ``If`` node's
        outputs. Names produced inside the branch are renamed to fresh,
        collision-free names; any other name resolves to an outer-scope tensor
        (subgraphs may capture the enclosing scope implicitly) and is kept
        as-is.
        """
        input_map = {
            vi.name: if_node.input[i]
            for i, vi in enumerate(branch.input)
            if i < len(if_node.input)
        }
        output_map = {vo.name: if_node.output[i] for i, vo in enumerate(branch.output)}
        local_names = {o for sub in branch.node for o in sub.output if o}
        local_names.update(init.name for init in branch.initializer)
        seen: dict[str, str] = {}
        taken = set(graph._out_to_node)  # pylint: disable=protected-access
        taken.update(i.name for i in graph.initializer)
        taken.update(graph.inputs)
        taken.update(graph.outputs)
        prefix = if_node.name or "if"

        def rename(old: str) -> str:
            if old == "":
                return old
            if old in input_map:
                return input_map[old]
            if old in output_map:
                return output_map[old]
            if old not in local_names:
                return old  # outer-scope reference, keep resolving to parent tensor
            if old not in seen:
                new = f"{prefix}/{old}"
                ind = 0
                while new in taken:
                    ind += 1
                    new = f"{prefix}/{old}_{ind}"
                taken.add(new)
                seen[old] = new
            return seen[old]

        for init in branch.initializer:
            new_init = deepcopy(init)
            new_init.name = rename(init.name)
            graph.initializer.append(new_init)

        for sub in branch.node:
            new_node = deepcopy(sub)
            self._rename_node(new_node, rename)
            new_node.name = f"{prefix}/{sub.name or sub.op_type}"
            self += new_node

    def _rename_node(self, node: NodeProto, rename) -> None:
        node.input[:] = [rename(i) for i in node.input]
        node.output[:] = [rename(o) for o in node.output]
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                self._rename_graph(attr.g, rename)
            elif attr.type == AttributeProto.GRAPHS:
                for g in attr.graphs:
                    self._rename_graph(g, rename)

    def _rename_graph(self, graph: GraphProto, rename) -> None:
        for vi in graph.input:
            if vi.name:
                vi.name = rename(vi.name)
        for vo in graph.output:
            if vo.name:
                vo.name = rename(vo.name)
        for node in graph.node:
            self._rename_node(node, rename)
