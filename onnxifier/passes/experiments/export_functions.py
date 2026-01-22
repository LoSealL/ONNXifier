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

# pylint: disable=arguments-differ

import itertools
import os
import shutil
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import networkx as nx
from onnx.onnx_pb import FunctionProto, NodeProto

from ... import OnnxGraph, logger
from ...utils import legalize_name
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import extract_function


@PASSES.register("export_functions", patch=["inline_local_functions"])
class ExportFunctionsRewriter(Rewriter):
    """This pass export all functions to plain onnx model.

    Args:
        save_dir (str | PathLike): Specify a directory to save exported functions.
            Functions will be saved in the topological order.
        optimize (bool): Optimize the exported functions.
        recursive (bool): Export functions recursively.
        namespace (str): Namespace of exported functions.

    Example:
        Model definition:
          Model(func1(func2(x)), func3(x))

        Files:
          - 00_func1.onnx
          - 00_00_func2.onnx (if recursive=True)
          - 01_func3.onnx
    """

    def __init__(
        self,
        save_dir: str | PathLike = ".",
        optimize: bool = True,
        recursive: bool = True,
        namespace: str = "",
    ):
        # With pre-order traversal to match nodes in topological order
        super().__init__(SingleNodePattern().with_domain("*").with_order("pre"))
        self.register_post_hook(self._post_hook)
        self.function_map: Dict[str, Tuple[int, OnnxGraph]] = {}
        self.path = Path(save_dir)
        self.optimize = optimize
        self.recursive = recursive
        self.namespace = namespace.strip("/")  # remove leading slash
        self._filelist: List[Path] = []

    def update_io_name(
        self,
        node: NodeProto,
        func: FunctionProto,
        func_nodes: List[NodeProto],
    ):
        """Update io names of nodes in function"""
        outside_io = {name for name in itertools.chain(node.input, node.output)}
        # change the io name which is the same as outside io
        # init: avoid func io is the same as node io
        io_rename = {name: name for name in itertools.chain(func.input, func.output)}
        for n in func_nodes:
            for node_io in itertools.chain(n.input, n.output):
                if node_io in outside_io and node_io not in io_rename:
                    io_rename[node_io] = f"{node_io}-{uuid4().hex[:16]}"

        # func.output may be the input of other nodes in func
        func_io_rename = {
            inside: outside
            for inside, outside in itertools.zip_longest(
                itertools.chain(func.input, func.output),
                itertools.chain(node.input, node.output),
            )
        }

        # update nodes io name
        for n in func_nodes:
            for j, node_input in enumerate(n.input):
                if node_input in io_rename:
                    node_input = io_rename[node_input]
                    n.input[j] = node_input
                if node_input in func_io_rename:
                    n.input[j] = func_io_rename[node_input]

        for n in func_nodes:
            for j, node_output in enumerate(n.output):
                if node_output in io_rename:
                    node_output = io_rename[node_output]
                    n.output[j] = node_output
                if node_output in func_io_rename:
                    n.output[j] = func_io_rename[node_output]

    def rewrite(
        self,
        graph: OnnxGraph,
        nodes: List[NodeProto],
        path: Optional[str | PathLike] = None,
    ):
        if path is not None:
            self.path = path
        node = nodes[0]
        if node.op_type not in graph.functions:
            raise RuntimeError(f"function {node.op_type} not found in the graph")

        model_name = node.name
        if model_name[0] == ".":
            model_name = model_name[1:]
        # prepend an ordered index to set exported file in the same order
        model_name = f"{len(self.function_map):02d}_{model_name}"
        if self.namespace:
            model_name = f"{self.namespace}_{model_name}"
        model_name = legalize_name(model_name)

        # step 1: find the input and output port that matches the custom node
        func = graph.functions[node.op_type]
        func_nodes = [deepcopy(node) for node in func.node]
        self.update_io_name(node, func, func_nodes)

        # step 2: recover constant name
        for func_node in func_nodes:
            if func_node.op_type == "Constant":
                func_node.name = func_node.output[0]

        # step 3: make a copy of each function node appended with identity names
        for i, n in enumerate(func_nodes):
            if not n.name:
                n.name = f"{uuid4().hex}"

        # step 3: assign attributes
        for attr in node.attribute:
            value = self.get_attribute(node, attr.name)
            for n in func_nodes:
                for attr2 in n.attribute:
                    if attr2.ref_attr_name == attr.name:
                        self.set_attribute(n, attr2.name, value)

        self.function_map[model_name] = len(self.function_map), extract_function(
            graph,
            func_nodes,
            # function name has been updated, no mapping needed.
            [(i, i) for i in node.input],
            [(i, i) for i in node.output],
        )

    def _post_hook(self, graph: OnnxGraph):
        for model_name, (order, subgraph) in self.function_map.items():
            from onnxifier import PassManager  # avoid cyclic import

            # cycle check
            cycles = list(nx.simple_cycles(subgraph))
            for cycle in cycles:
                logger.error(f"cycle: {cycle}")
            if self.optimize:
                pm = PassManager(["fold_constant", "eliminate_dead_nodes"])
                subgraph = pm.optimize(subgraph, strict=False)
            if self.recursive and subgraph.functions:
                rewrite_fn = ExportFunctionsRewriter(
                    self.path,
                    self.optimize,
                    self.recursive,
                    f"{self.namespace}/{order:02d}",
                )
                rewrite_fn(subgraph)
                self._filelist.extend(rewrite_fn.filelist)
            else:
                subgraph.save(f"{self.path}/{model_name}.onnx")
                self._filelist.append(Path(f"{self.path}/{model_name}.onnx"))
        # copy external data
        for data in graph.external_data:
            data_path = Path(self.path) / data.name
            if data_path.exists():
                continue
            try:
                os.symlink(data, data_path)
            except (OSError, PermissionError):
                shutil.copyfile(data, data_path)
        return graph

    @property
    def filelist(self) -> List[Path]:
        """List of extracted function files."""
        return self._filelist
