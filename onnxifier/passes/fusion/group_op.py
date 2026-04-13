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

from collections.abc import Mapping, MutableMapping

import networkx as nx
from onnx import NodeProto
from onnx.helper import make_function, make_node, make_operatorsetid

from ...graph import OnnxGraph
from ...logger import debug
from .. import PASSES

DOMAIN = "org.onnxifier"
type HierType = Mapping[str, str | HierType]
type MutableHierType = MutableMapping[str, str | MutableHierType]


def _is_unique_constant(graph: OnnxGraph, node: NodeProto) -> bool:
    """Check if `node` is a Constant node that feeds into only one other node."""
    if node.op_type != "Constant":
        return False
    successors = list(graph.onnx_successors(node))
    return len(successors) == 1


def _make_group(
    graph: OnnxGraph,
    nodes: list[NodeProto],
    group_name: str,
    force: bool = False,
) -> NodeProto:
    """Create an ONNX function from `nodes`, add the call-node to `graph`,
    remove the original nodes, and return the call-node."""
    h = graph.onnx_subgraph(nodes)
    h.name = group_name
    func = make_function(
        DOMAIN,
        group_name,
        inputs=[i.name for i in h.input],
        outputs=[i.name for i in h.output],
        nodes=nodes,
        opset_imports=[
            make_operatorsetid(DOMAIN, graph.opset_version),
            make_operatorsetid("", graph.opset_version),
        ],
    )
    graph.onnx_add_function(func, force=force)
    hnode = make_node(
        op_type=group_name,
        inputs=[i.name for i in h.input],
        outputs=[i.name for i in h.output],
        name=group_name,
        domain=DOMAIN,
    )
    graph.add_onnx_node(hnode)
    for n in nodes:
        graph.remove_onnx_node(n)
    return hnode


def _group_operators_recursively(
    graph: OnnxGraph, hier: HierType, group_name: str
) -> list[NodeProto]:
    nodes_set = []
    for name, hierarchy in hier.items():
        if isinstance(hierarchy, Mapping):
            # subgraph node
            nodes = _group_operators_recursively(graph, hierarchy, name)
        else:
            nodes = [graph.nodes[hierarchy]["pb"]]
        # fuse constants
        for node in nodes:
            for pred in graph.onnx_predecessors(node):
                if _is_unique_constant(graph, pred):
                    nodes_set.append(pred)
            nodes_set.append(node)

    # Split disconnected components into separate groups
    h = graph.onnx_subgraph(nodes_set)
    components = sorted(nx.weakly_connected_components(h), key=min)
    if len(components) > 1:
        grouped_nodes = []
        for idx, component in enumerate(components):
            component_nodes = [graph.nodes[n]["pb"] for n in component if n in graph]
            grouped = _make_group(graph, component_nodes, f"{group_name}_{idx}")
            grouped_nodes.append(grouped)
        return grouped_nodes

    return [_make_group(graph, nodes_set, group_name)]


def _merge_cycles(graph: OnnxGraph, max_iters: int = 100):
    def _merged_group_name(component: set[str]) -> str:
        candidates: list[tuple[str, int]] = []
        for node_name in component:
            call_node = graph.nodes[node_name]["pb"]
            func_name = call_node.op_type
            func = graph.functions.get(func_name)
            candidates.append((func_name, len(func.node) if func is not None else 1))
        return min(candidates, key=lambda item: (-item[1], item[0]))[0]

    def _find_cyclic_component() -> set[str] | None:
        for comp in nx.strongly_connected_components(graph):
            if len(comp) > 1:
                return set(comp)
            [node_name] = list(comp)
            if graph.has_edge(node_name, node_name):
                return {node_name}
        return None

    step = 0
    cyc_comp = _find_cyclic_component()
    while cyc_comp is not None and step < max_iters:
        merged_name = _merged_group_name(cyc_comp)
        debug(f"grouped graph has cycles: {cyc_comp}->{merged_name}, step={step}")
        step += 1
        # Collect the underlying original operator/call nodes from each cyclic
        # call node by unfolding one level of the function library.
        body_nodes: list[NodeProto] = []
        for node_name in cyc_comp:
            call_node = graph.nodes[node_name]["pb"]
            func_name = call_node.op_type
            if func_name in graph.functions:
                body_nodes.extend(graph.functions[func_name].node)
            else:
                body_nodes.append(call_node)

        # Remove the cyclic call nodes from the live graph.
        for node_name in cyc_comp:
            graph.remove_onnx_node(graph.nodes[node_name]["pb"])

        # Re-insert the collected body nodes so the graph is a DAG again,
        # then fuse them into a single new function.
        dedup_body_nodes = {n.name: n for n in body_nodes}
        for node in dedup_body_nodes.values():
            graph.add_onnx_node(node)

        # merge constants
        for node in list(dedup_body_nodes.values()):
            for pred in graph.onnx_predecessors(node):
                if _is_unique_constant(graph, pred):
                    dedup_body_nodes[pred.name] = pred
        _make_group(
            graph,
            list(dedup_body_nodes.values()),
            merged_name,
            force=merged_name in graph.functions,
        )
        cyc_comp = _find_cyclic_component()

    if cyc_comp is not None:
        raise RuntimeError(
            f"_merge_cycles did not converge within max_iters={max_iters}."
        )


def _prune_unused_functions(graph: OnnxGraph) -> None:
    reachable: set[str] = set()
    stack = [graph.nodes[name]["pb"].op_type for name in graph]

    while stack:
        func_name = stack.pop()
        if func_name in reachable or func_name not in graph.functions:
            continue
        reachable.add(func_name)
        stack.extend(node.op_type for node in graph.functions[func_name].node)

    removable = [
        name
        for name, func in graph.functions.items()
        if func.domain == DOMAIN and name not in reachable
    ]
    for name in removable:
        del graph.functions[name]


@PASSES.register("group", deps=["initializer_to_constant"])
def group_operators(graph: OnnxGraph, depth: int = 1, sep: str = "/") -> OnnxGraph:
    """Group operators into nested subgraphs (functions) based on their name.

    Note:

        Prefix is obtained by splitting name by `sep` and taking first `depth` parts.

    Example:

        model/backbone/layer1/conv1
            -> model -> /backbone/layer1/conv1 (depth=1)
            -> model -> /backbone -> /layer1/conv1 (depth=2)
            -> model -> /backbone -> /layer1 -> /conv1 (depth>=3)
    """

    assert depth >= 1 and len(sep) == 1
    hier: MutableHierType = {}
    for node in graph:
        node_pb = graph.nodes[node]["pb"]
        if not node_pb.HasField("name") or node_pb.op_type == "Constant":
            continue
        name = node_pb.name.strip(sep)  # remove leading and trailing slashes
        if sep not in name:
            continue
        # remove leading and trailing slashes, then split by sep
        *parents, leaf = name.split(sep, depth)

        curr = hier
        for parent in parents:
            # Detect naming collisions where a path segment was previously used
            # as a leaf node (string) but now needs to become a subtree.
            if isinstance(curr, str):
                raise ValueError(
                    f"Naming collision in group_operators: path segment '{parent}' "
                    f"was previously used as a leaf node when processing '{name}'."
                )
            if parent not in curr:
                curr[parent] = {}
            elif isinstance(curr[parent], str):
                # Existing leaf at this path cannot be converted into a subtree
                # without changing semantics; report a clear error instead.
                full_path = sep.join((*parents[: parents.index(parent) + 1],))
                raise ValueError(
                    "Naming collision in group_operators: path segment "
                    f"'{full_path}' is used both as an operator name and as a "
                    f"prefix for other operators (while processing '{name}')."
                )
            curr = curr[parent]
        assert isinstance(curr, dict)
        curr[leaf] = node

    for name, hierarchy in hier.items():
        if isinstance(hierarchy, Mapping):
            _group_operators_recursively(graph, hierarchy, name)

    _merge_cycles(graph)
    _prune_unused_functions(graph)
    return graph
