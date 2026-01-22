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

import re
from abc import ABCMeta, abstractmethod
from collections.abc import Sized
from itertools import chain, product
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
from onnx import AttributeProto, NodeProto
from onnx.helper import get_attribute_value, make_attribute

from ..graph import OnnxGraph
from .utils import attribute_value


class Pattern(metaclass=ABCMeta):
    """Pattern to be matched. A pattern can be ORed and ADDed.

    Example::

        p1 = SingleNodePattern("Conv")
        p2 = SingleNodePattern("Relu")
        conv_or_relu = p1 | p2  # same as `p1 + p2`
    """

    @abstractmethod
    def match(
        self,
        graph: OnnxGraph,
        specify_node_names: Set[str] | None = None,
    ) -> Iterable[NodeProto | List[NodeProto]]:
        """Implementation how to match in the graph

        Args:
            graph (OnnxGraph): onnx graph
            specify_node_names (set[str], optional): Only optimize nodes with these
                names. Defaults to None.
        """

    def __or__(self, pattern: "Pattern") -> "Pattern":
        return OrPattern(self, pattern)

    def __add__(self, pattern: "Pattern") -> "Pattern":
        return OrPattern(self, pattern)

    def __radd__(self, pattern) -> "Pattern":
        """To support builtin functions like `sum`"""
        if isinstance(pattern, int):
            return self
        return OrPattern(self, pattern)


class OrPattern(Pattern):
    """A special pattern to match either p1 or p2."""

    def __init__(self, p1: Pattern, p2: Pattern):
        self.patterns: List[Pattern] = []
        if isinstance(p1, OrPattern):
            self.patterns.extend(p1.patterns)
        else:
            self.patterns.append(p1)
        if isinstance(p2, OrPattern):
            self.patterns.extend(p2.patterns)
        else:
            self.patterns.append(p2)

    def match(self, graph: OnnxGraph, specify_node_names: Set[str] | None = None):
        yield from chain(*(p.match(graph, specify_node_names) for p in self.patterns))


class SingleNodePattern(Pattern):
    """Match a single node type."""

    __id__ = 0

    def __init__(self, op_type: Optional[str] = None, op_name: Optional[str] = None):
        self.op_type = op_type
        self.op_name = op_name
        self.attr: List[AttributeProto] = []
        self.inputs: Optional[Sequence[str | None]] = None
        self.outputs: Optional[Sequence[str | None]] = None
        self.domain: Optional[str] = None
        self.order: Optional[Literal["pre", "post"]] = None
        self.id = SingleNodePattern.__id__
        SingleNodePattern.__id__ += 1

    def match(self, graph: OnnxGraph, specify_node_names: Set[str] | None = None):
        if isinstance(graph, OnnxGraph):
            if self.order == "pre":
                node_gen = nx.topological_sort(graph)
            elif self.order == "post":
                node_gen = nx.topological_sort(graph.reverse(copy=False))
            else:
                node_gen = iter(graph)
            for node in node_gen:
                node_pb = graph.nodes[node]["pb"]
                if self._check(node_pb, graph, specify_node_names):
                    yield node_pb
        else:  # match a single node
            node_pb = graph
            if self._check(node_pb, None, specify_node_names):
                yield node_pb

    def _check(self, node, graph, node_names):
        conditions = [
            self._check_type(node, self.op_type),
            self._check_attr(node),
            self._check_inputs(node, graph),
            self._check_outputs(node, graph),
            self._check_domain(node),
        ]
        if self.op_name:
            conditions.append(self._check_name(node, {self.op_name}))
        elif node_names:
            conditions.append(self._check_name(node, node_names))
        return all(conditions)

    def _check_type(self, node: NodeProto, op_type: str | None):
        return op_type == node.op_type or op_type is None

    def _check_name(self, node: NodeProto, names: Set[str]):
        return not names or node.name in names

    def _check_attr(self, node: NodeProto):
        if not self.attr:
            return True
        match_table: Dict[str, AttributeProto] = {}  # all attribute to check
        for attr in self.attr:
            match_table[attr.name] = attr
        matched = 0  # matched number of attributes
        for attr in node.attribute:
            if attr.name in match_table:
                matched += 1 if self._match_attr(match_table[attr.name], attr) else 0
        return matched == len(match_table)

    def _match_attr(self, attr0: AttributeProto, attr1: AttributeProto):
        def _is_any(v):
            if v.type == AttributeProto.STRING:
                return get_attribute_value(v).decode() == "__ANY__"
            return False

        if _is_any(attr0) or _is_any(attr1):
            return True
        if attr0.type != attr1.type:
            return False
        return attribute_value(attr0) == attribute_value(attr1)

    def _check_inputs(self, node: NodeProto, graph: OnnxGraph):
        if self.inputs is None or graph is None:
            return True
        input_nodes = [i.name for i in graph.onnx_predecessors(node)]
        for i in filter(lambda i: i.name in node.input, graph.initializer):
            input_nodes.append(i.name)
        if len(input_nodes) != len(self.inputs):
            return False
        return all(i is None or i in set(input_nodes) for i in self.inputs)

    def _check_outputs(self, node: NodeProto, graph: OnnxGraph):
        if self.outputs is None or graph is None:
            return True
        output_nodes = [i.name for i in graph.onnx_successors(node)]
        if len(output_nodes) != len(self.outputs):
            return False
        return all(i is None or i in set(output_nodes) for i in self.outputs)

    def _check_domain(self, node: NodeProto):
        if not self.domain:
            # do not compare if self.domain is not specified
            return self.domain is None or node.domain in ("", "ai.onnx")
        if "*" in self.domain or "?" in self.domain:
            # do not match ai.onnx
            if node.domain in ("", "ai.onnx"):
                return False
            pattern = re.compile(self.domain.replace("*", ".*").replace("?", "."))
            if re.findall(pattern, node.domain):
                return True
            return False
        return self.domain == node.domain

    def __hash__(self):
        op_type = self.op_type or ""
        op_name = self.op_name or ""
        return hash(f"{op_type}{op_name}{self.id}")

    def with_attr(
        self, name: str | AttributeProto, value: Optional[Any] = None
    ) -> "SingleNodePattern":
        """Match the node with specific attribute."""
        if isinstance(name, AttributeProto):
            self.attr.append(name)
        elif value is not None:
            self.attr.append(make_attribute(name, value))
        else:
            self.attr.append(make_attribute(name, "__ANY__"))
        return self

    def with_name(self, name: str) -> "SingleNodePattern":
        """Match the node with specific name."""
        assert isinstance(name, str)
        self.op_name = name
        return self

    def with_inputs(self, num_inputs: int, *input_nodes: str) -> "SingleNodePattern":
        """Match the node with specific number of inputs and names.

        Example::

            # a concat with 2 inputs
            SingleNodePattern("Concat").with_inputs(2)

            # an add with an input from node "X"
            SingleNodePattern("Add").with_inputs(2, "X", None)

        Args:
            num_inputs (int): number of inputs
            *input_nodes (str): names of each input node, this is not order sensitive,
                you can use `None` to match any input.
        """
        if input_nodes:
            assert len(input_nodes) == num_inputs
            self.inputs = list(input_nodes)
        else:
            self.inputs = [None] * num_inputs
        return self

    def with_outputs(self, num_outputs: int, *output_nodes: str) -> "SingleNodePattern":
        """Match the node with specific number of outputs and names.

        Example::

            # a split with 2 outputs
            SingleNodePattern("Split").with_outputs(2)

        Args:
            num_outputs (int): number of outputs
            *output_nodes (str): names of each output node, this is not order sensitive,
                you can use `None` to match any output.
        """
        if output_nodes:
            assert len(output_nodes) == num_outputs
            self.outputs = list(output_nodes)
        else:
            self.outputs = [None] * num_outputs
        return self

    def with_domain(self, domain: str) -> "SingleNodePattern":
        """Match the node with specific domain.

        Example::

            # a node with custom domain "onnxifier"
            SingleNodePattern().with_domain("onnxifier")

        Args:
            domain (str): domain name, if domain name is a wildcard
                (supports * and ? for now), match any domain using
                regex rules.
        """
        self.domain = str(domain)
        return self

    def with_order(self, order: Literal["pre", "post"]) -> "SingleNodePattern":
        """Match the node in specific order.

        Args:
            order (str):
            - "pre" to match the node in topological order.
            - "post" to match the node in reverse topological order.
        """
        self.order = order
        return self


class GraphPattern(Pattern, nx.DiGraph):
    """Match a subgraph."""

    def __init__(self, dag: Optional[nx.DiGraph] = None):
        if dag is not None:
            assert isinstance(dag, nx.DiGraph)
            assert all(isinstance(n, SingleNodePattern) for n in dag.nodes)
        super().__init__(dag)

    def add_node(self, node_for_adding, **attr) -> "GraphPattern":  # type: ignore
        assert isinstance(node_for_adding, Pattern)
        attr.update(pattern=node_for_adding)
        super().add_node(node_for_adding, **attr)
        return self

    def add_edge(self, u_of_edge, v_of_edge, **attr) -> "GraphPattern":  # type: ignore
        assert isinstance(u_of_edge, Pattern)
        assert isinstance(v_of_edge, Pattern)
        self.add_node(u_of_edge)
        self.add_node(v_of_edge)
        super().add_edge(u_of_edge, v_of_edge, **attr)
        return self

    def match(self, graph: OnnxGraph, specify_node_names: Set[str] | None = None):
        # Matching subgraph using nx method
        def _nm(g, h):
            return list(h["pattern"].match(g["pb"]))

        matcher = DiGraphMatcher(graph, self, node_match=_nm)
        for matches in matcher.subgraph_isomorphisms_iter():
            # match specify_node_names by "OR" pattern
            if specify_node_names:
                if any(key in specify_node_names for key in matches.keys()):
                    yield [graph.nodes[key]["pb"] for key in matches.keys()]
            else:
                yield [graph.nodes[key]["pb"] for key in matches.keys()]


class ConstantGraphPattern(Pattern):
    """Match each subgraph contains constant nodes"""

    def _is_deterministic(self, node) -> bool:
        return node.op_type not in {
            "RandomUniform",
            "RandomNormal",
            "RandomUniformLike",
            "RandomNormalLike",
            "Multinomial",
        }

    def _is_qdq(self, node) -> bool:
        return node.op_type in {
            "DequantizeLinear",
            "DynamicQuantizeLinear",
            "QuantizeLinear",
        }

    def _has_subgraph(self, node) -> bool:
        return any(
            attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)
            for attr in node.attribute
        )

    def match(self, graph: OnnxGraph, specify_node_names: Set[str] | None = None):
        const_names = {None, ""}  # for inputs with empty (dangled) name
        # get from initializers
        const_names.update(i.name for i in graph.initializer)
        const_nodes = []
        for _, node_name in enumerate(nx.topological_sort(graph)):
            node = graph.nodes[node_name]["pb"]
            if not self._is_deterministic(node) or self._is_qdq(node):
                continue
            if self._has_subgraph(node):
                continue
            if all(i in const_names for i in node.input):
                const_nodes.append(node)
                const_names.update(node.output)
        if specify_node_names:
            const_nodes = [n for n in const_nodes if n.name in specify_node_names]
        # TODO: cluster into subgraphs
        yield const_nodes


class OutputNodePattern(Pattern):
    """Match output nodes

    Args:
        match_all (bool): whether to match all output nodes or just one of them.
    """

    def __init__(self, match_all=False):
        self.match_all = match_all

    def match(self, graph: OnnxGraph, specify_node_names: Set[str] | None = None):
        nodes = []
        for output in graph.output:
            for node_name in reversed(list(nx.topological_sort(graph))):
                node = graph.nodes[node_name]["pb"]
                if output.name in node.output:
                    if self.match_all:
                        nodes.append(node)
                    elif not specify_node_names or node.name in specify_node_names:
                        yield node
        if self.match_all:
            if specify_node_names:
                nodes = [n for n in nodes if n.name in specify_node_names]
            yield nodes


class InputNodePattern(Pattern):
    """Match input nodes

    Args:
        match_all (bool): whether to match all output nodes or just one of them.
    """

    def __init__(self, match_all=False):
        self.match_all = match_all

    def match(self, graph: OnnxGraph, specify_node_names: Set[str] | None = None):
        nodes = []
        for input_var in graph.input:
            for node_name in nx.topological_sort(graph):
                node = graph.nodes[node_name]["pb"]
                if input_var.name in node.input:
                    if self.match_all:
                        nodes.append(node)
                    elif not specify_node_names or node.name in specify_node_names:
                        yield node
        if self.match_all:
            if specify_node_names:
                nodes = [n for n in nodes if n.name in specify_node_names]
            yield nodes


class StartEndPointPattern(Pattern):
    """Match a start-relay-end point pattern."""

    def __init__(self, start_pattern: Pattern, end_pattern: Pattern):
        self.start = start_pattern
        self.end = end_pattern

    def match(self, graph: OnnxGraph, specify_node_names: Set[str] | None = None):
        for beg, end in product(
            self.start.match(graph, specify_node_names),
            self.end.match(graph, specify_node_names),
        ):
            g = graph.copy(as_view=False)
            assert isinstance(g, OnnxGraph)
            if isinstance(beg, Sized) and len(beg) > 1:
                beg_node = self._merge_subgraph(g, beg)
            else:
                beg = beg[0] if isinstance(beg, Sequence) else beg
                beg_node = beg.name
            if isinstance(end, Sized) and len(end) > 1:
                end_node = self._merge_subgraph(g, end)
            else:
                end = end[0] if isinstance(end, Sequence) else end
                end_node = end.name
            # search for a path from start to end
            matched_nodes: Set[NodeProto] = set()
            try:
                paths = nx.shortest_simple_paths(g, beg_node, end_node)
                for i in [node for nodes in paths for node in nodes]:
                    if isinstance(i, nx.DiGraph):
                        matched_nodes.update(i.nodes[j]["pb"] for j in i)
                    else:
                        matched_nodes.add(graph.nodes[i]["pb"])
            except nx.NodeNotFound:
                continue
            yield list(matched_nodes)

    def _merge_subgraph(self, graph: nx.DiGraph, nodes):
        """Merge subgraph into a single node."""
        h: nx.DiGraph = graph.subgraph([n.name for n in nodes]).copy(  # type: ignore
            as_view=False,
        )
        pred_nodes: Set[NodeProto] = set()
        for in_node in filter(lambda i: h.in_degree(i) == 0, h):
            pred_nodes.update(graph.predecessors(in_node))
        succ_nodes: Set[NodeProto] = set()
        for out_node in filter(lambda o: h.out_degree(o) == 0, h):
            succ_nodes.update(graph.successors(out_node))
        graph.remove_nodes_from([n.name for n in nodes])
        for i in pred_nodes:
            graph.add_edge(i, h)
        for i in succ_nodes:
            graph.add_edge(h, i)
        return h
