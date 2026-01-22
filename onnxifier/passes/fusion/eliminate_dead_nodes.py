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

import networkx as nx

from ... import logger
from ...graph import OnnxGraph
from .. import L2


@L2.register()
def eliminate_dead_nodes(graph: OnnxGraph):
    """Remove nodes who doesn't connect to any output"""

    dead_nodes = set()
    exit_nodes = set(
        [graph.nodes[i]["pb"].name for i in graph.nodes if graph.nodes[i]["has_output"]]
    )
    for i in nx.topological_sort(graph.reverse()):
        if not any(nx.has_path(graph, i, j) for j in exit_nodes):
            dead_nodes.add(graph.nodes[i]["pb"])
            logger.debug(f"Dead node found: {i}")
    for node in dead_nodes:
        graph.remove_onnx_node(node)
    return graph
