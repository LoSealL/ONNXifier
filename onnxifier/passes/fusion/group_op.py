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

from typing import Mapping, MutableMapping, TypeAlias, Union

from onnx import NodeProto
from onnx.helper import make_function, make_node, make_operatorsetid

from ...graph import OnnxGraph
from .. import PASSES

DOMAIN = "org.onnxifier"
HierType: TypeAlias = Mapping[str, Union[str, "HierType"]]
MutableHierType: TypeAlias = MutableMapping[str, Union[str, "MutableHierType"]]


def _group_operators_recursively(
    graph: OnnxGraph, hier: HierType, group_name: str
) -> NodeProto:
    nodes = []
    for name, hierarchy in hier.items():
        if isinstance(hierarchy, Mapping):
            # subgraph node
            node = _group_operators_recursively(graph, hierarchy, name)
        else:
            node = graph.nodes[hierarchy]["pb"]
        # fuse constants
        for pred in graph.onnx_predecessors(node):
            if pred.op_type == "Constant":
                nodes.append(pred)
        nodes.append(node)
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
    graph.onnx_add_function(func)
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

    return graph
