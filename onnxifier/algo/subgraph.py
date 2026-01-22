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

from typing import Iterable, List, Optional, Tuple

from onnx import NodeProto

from ..graph import OnnxGraph


def find_inbound_nodes(
    graph: OnnxGraph, node: NodeProto, startpoints: Iterable[str]
) -> Tuple[List[NodeProto], List[NodeProto]]:
    r"""Find a set of predecessors of the given node and all of them have a path
    to the given node while the path doesn't contain any node in the given set
    ``startpoints``.

    Example:

           A(Conv)    B(Conv)
             |           |
           C(Relu)       |
             \           /
              \         /
               D(Concat)
                   |
               E(Sigmoid)
                   |
                F(Conv)

        find_inbound_nodes(g, F, ["Conv"]) == ([C, D, E], [A, B])

    Args:
        graph (OnnxGraph): The ONNX graph.
        node (NodeProto): The node to find predecessors for.
        startpoints (Iterable[str]): A set of node names that are not allowed to be
            in the path to the given node.

    Returns:
        Tuple[List[NodeProto]]: A list of found nodes, and a list of start nodes.
    """

    preds = list(graph.onnx_predecessors(node))
    inbounds = []
    start_nodes = []
    while preds:
        pred = preds.pop(0)
        if pred.op_type in set(startpoints):
            start_nodes.append(pred)
        else:
            inbounds.append(pred)
            preds.extend(graph.onnx_predecessors(pred))
    return inbounds, start_nodes


def find_outbound_nodes(
    graph: OnnxGraph, node: NodeProto, endpoints: Iterable[str]
) -> Tuple[List[NodeProto], List[NodeProto]]:
    r"""Find a set of successors of the given node and all of them have a path
    to the given node while the path doesn't contain any node in the given set
    ``endpoints``.

    Example:

           A(Conv)    B(Conv)
             |           |
           C(Relu)       |
             \           /
              \         /
               D(Concat)
                   |
               E(Sigmoid)
                   |
                F(Conv)

        find_outbound_nodes(g, A, ["Conv"]) == ([C, D, E], [F])

    Args:
        graph (OnnxGraph): The ONNX graph.
        node (NodeProto): The node to find successors for.
        endpoints (Iterable[str]): A set of node names that are not allowed to be
            in the path to the given node.

    Returns:
        Tuple[List[NodeProto]]: A list of found nodes, and a list of end nodes.
    """

    succs = list(graph.onnx_successors(node))
    outbounds = []
    end_nodes = []
    while succs:
        succ = succs.pop(0)
        if succ.op_type in set(endpoints):
            end_nodes.append(succ)
        else:
            outbounds.append(succ)
            succs.extend(graph.onnx_successors(succ))
    return outbounds, end_nodes


def find_sibling_nodes(
    graph: OnnxGraph,
    node: NodeProto,
    siblings: Optional[Iterable[str]] = None,
    parents: Optional[Iterable[str]] = None,
) -> List[NodeProto]:
    r"""Find the sibling nodes of the given node in the graph.

    A sibling node is a node that share the same parent node with the given one.

    Example:

             A (Conv)
            /       \
           /         \
        B (Conv)   C (Conv)

        find_sibling_nodes(g, B, ["Conv"]) == [C]

    Args:
        graph (OnnxGraph): a DAG graph.
        node (NodeProto): the given node to find its sibling nodes.
        siblings (Iterable[str], optional): a filter of types to regard as siblings. If
            not provided, the node with same type as the given node are regarded as
            siblings. Defaults to None.
        parents (Iterable[str], optional): a filter of types to regard as
            parents. If not provided, the first predecessor node is regarded as its
            parent. Defaults to None.

    Returns:
        List[NodeProto]: a list of sibling nodes.
    """

    if parents is None:
        parents = [op.op_type for op in graph.onnx_predecessors(node)]
    if siblings is None:
        siblings = [node.op_type]
    else:
        siblings = list(siblings) + [node.op_type]

    _, parent_nodes = find_inbound_nodes(graph, node, parents)
    sibling_nodes = set()
    for n in parent_nodes:
        _, siblings_under_n = find_outbound_nodes(graph, n, siblings)
        sibling_nodes.update(siblings_under_n)

    return list(sibling_nodes.difference([node]))
