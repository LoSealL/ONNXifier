"""
Copyright (C) 2024-2025 The ONNXIFIER Authors.

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

from typing import List, Sequence

import networkx as nx

from ... import logger
from ...graph import OnnxGraph
from ...passes import PASSES
from . import OP_CONVERTER


def _expand_deps(deps: Sequence[str]):
    root: nx.DiGraph = nx.DiGraph()
    root.add_nodes_from(deps)
    leaves: List[str] = list(deps).copy()
    while leaves:
        leaf = leaves.pop(0)
        children = PASSES[leaf].__DEPS__
        leaves.extend(children)
        root.add_edges_from([(leaf, child) for child in children])
        try:
            cycles = nx.find_cycle(root, leaf)
        except nx.NetworkXNoCycle:
            continue
        else:
            logger.error(f"Cyclic dependencies found!: {cycles}")
            break
    yield from reversed(list(nx.topological_sort(root)))


def downgrade_op_version(graph: OnnxGraph, op_version: int = 17):
    """Downgrade the op version of all nodes in the graph to the specified version."""

    if op_version not in OP_CONVERTER:
        raise NotImplementedError(
            f"Support to downgrade to {sorted(OP_CONVERTER.keys())} for now, "
            f"got {op_version}."
        )
    if graph.opset_version <= op_version:
        return graph
    version_steps = sorted(OP_CONVERTER.keys(), reverse=True)
    for v1, v0 in zip(version_steps, version_steps[1:]):
        if graph.opset_version > v1 and op_version <= v0:
            graph = downgrade_op_version(graph, v1)
            return downgrade_op_version(graph, op_version)

    logger.debug(f"downgrading opset to version {op_version}")
    node_types = set(node["pb"].op_type for node in graph.nodes.values())
    for node_type in node_types:
        if converter := OP_CONVERTER[op_version].get(node_type):
            try:
                for deps in _expand_deps(converter.__DEPS__):
                    graph = PASSES[deps](graph)
                graph = converter(graph)
            except Exception:  # pylint: disable=broad-except
                logger.error(f"Failed to downgrade {node_type} to {op_version}")
                raise
    graph.opset_version = op_version
    return graph
