"""
Copyright (C) 2024-2026 The ONNXIFIER Authors.

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

import inspect
import traceback
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence, Set
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Optional

import networkx as nx
from onnx import NodeProto, ValueInfoProto
from onnx.helper import make_function, make_value_info
from tabulate import tabulate
from termcolor import colored

from .graph import OnnxGraph
from .logger import debug, error, warning
from .passes import L1, L2, L3, PASSES
from .passes.utils import extract_function
from .traits import RewriterInterface


def _is_function(rewriter) -> bool:
    """Return True when ``rewriter`` is either a plain function or a
    ``functools.partial`` object that wraps a function.

    This simplifies repeated inline checks like:
    ``(isinstance(rewriter, partial) and
    inspect.isfunction(rewriter.func)) or inspect.isfunction(rewriter)``
    and centralises the logic for easier testing and future changes.
    """
    is_partial_func = False
    if isinstance(rewriter, partial) and inspect.isfunction(rewriter.func):
        is_partial_func = True
    is_plain_func = inspect.isfunction(rewriter)
    return is_partial_func or is_plain_func


def _apply_pass(rewriter, g, specify_node_names):
    if _is_function(rewriter) or not specify_node_names:
        g = rewriter(g)
    else:
        g = rewriter(g, _specify_node_names=specify_node_names)
    return g


class PassManager:
    """Ordered optimization pass list.

    Args:
        include (List[str | RewriterInterface], Optional): a list of pattern to
            select passes. Defaults to select L1, L2 and L3 passes.
        exclude (List[str], Optional): a list of pattern to deselect passes.
            Defaults to None.
        configs (Dict[str, Any], Optional): a dictionary of pass configurations.
            Defaults to None.
    """

    def __init__(
        self,
        include: Optional[Sequence[str | RewriterInterface]] = None,
        exclude: Optional[Sequence[str]] = None,
        configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        passes: List[str | RewriterInterface]
        if include is None:
            passes = [i for i in chain(L1, L2, L3)]
        else:
            passes = [i for i in include]
        if exclude:
            passes = list(filter(lambda i: i not in exclude, passes))
        activated: List[RewriterInterface] = []
        for i in passes:
            if isinstance(i, str):
                if i in PASSES:
                    activated.append(PASSES[i])
                else:
                    warning(f"{i} is not registered as a pass, ignore it.")
            else:
                activated.append(i)
        self.activated: List[RewriterInterface] = activated
        if configs:
            self._assign_config_to_pass(configs)

    def _assign_config_to_pass(self, configs: Dict[str, Any]):
        for key, config in configs.items():
            index = -1
            if ":" in key:
                key, index_str = key.split(":", 2)
                index = int(index_str)
            if not isinstance(config, Mapping):
                warning(f"config {key}:{index} must be a dict, but got {type(config)}")
                continue
            candidates = [i for i in self.activated if i.__NAME__ == key]
            if index >= 0 and index >= len(candidates):
                warning(
                    f"config {key}:{index} exceeds the boundary. "
                    f"Number of {key} is {len(candidates)}"
                )
                continue
            if index >= 0:
                candidates = [candidates[index]]
            for func in candidates:
                pos = self.activated.index(func)
                self.activated[pos] = partial(func, **config)  # type: ignore
                self.activated[pos].__NAME__ = key
                self.activated[pos].__DEPS__ = func.__DEPS__
                self.activated[pos].__PATCHES__ = func.__PATCHES__

    def _expand(self, nodes, priv_member):
        root: nx.DiGraph = nx.DiGraph()  # type: ignore
        leaves = [f"{i}:{n}" for i, n in enumerate(nodes)]
        root.add_nodes_from(leaves)
        # a shallow graph to check cyclic dependencies
        shallow: nx.DiGraph = nx.DiGraph()  # type: ignore
        shallow.add_nodes_from(nodes)
        ind = len(leaves)
        while leaves:
            leaf = leaves.pop(0)
            leaf_pass = leaf.split(":")[1]
            children = getattr(PASSES.get(leaf_pass), priv_member, [])
            shallow.add_edges_from([(leaf_pass, child) for child in children])
            try:
                cycles = nx.find_cycle(shallow, leaf_pass)
            except nx.NetworkXNoCycle:
                children = [f"{ind + i}:{c}" for i, c in enumerate(children)]
                ind += len(children)
                root.add_edges_from([(leaf, child) for child in children])
                leaves.extend(children)
            else:
                error(f"Cyclic dependencies found!: {cycles}")
                raise RuntimeError("Cyclic dependencies found!")
        return root

    def _expand_deps(self, deps):
        root = self._expand(deps, "__DEPS__")
        for i in nx.traversal.dfs_postorder_nodes(root):
            yield i.split(":")[1]

    def _expand_patches(self, nodes):
        root = self._expand(nodes, "__PATCHES__")
        for i in nx.traversal.dfs_preorder_nodes(root):
            yield i.split(":")[1]

    def optimize(
        self,
        graph: OnnxGraph,
        strict: bool = False,
        recursive: bool = False,
        specify_node_names: Optional[Set[str]] = None,
    ) -> OnnxGraph:
        """Invoke passes on the input graph.

        Args:
            graph (OnnxGraph): See :class:`OnnxGraph`.
            strict (bool): Break if any pass fails.
            recursive (bool): Recursively apply passes to functions.
            specify_node_names (set[str], optional): Only optimize nodes with these
                names. Defaults to None.
        """
        graphs = {"": ("", graph)}  # let "" be the main graph
        if recursive:
            graphs.update(self._make_subgraph_from_functions(graph))
        for opt in self.activated:
            for name, (d, g) in graphs.items():
                try:
                    for deps in self._expand_deps(opt.__DEPS__):
                        if rewriter := PASSES.get(deps):
                            debug(f"Applying dependency pass: {deps} to {name}")
                            g = _apply_pass(rewriter, g, specify_node_names)
                    debug(f"Applying pass: {opt.__NAME__} to {name}")
                    g = _apply_pass(opt, g, specify_node_names)
                    for patch in self._expand_patches(opt.__PATCHES__):
                        debug(f"Applying patch pass: {patch} to {name}")
                        if rewriter := PASSES.get(patch):
                            g = _apply_pass(rewriter, g, specify_node_names)
                except Exception as ex:  # pylint: disable=broad-exception-caught
                    error(f"{opt.__NAME__} failed at {name}: {ex}")
                    debug("\n".join(traceback.format_exception(ex)))
                    if strict:
                        raise
                graphs[name] = (d, g)
                if name:
                    function = self._make_function_from_subgraph(name, d, g)
                    graphs[""][1].onnx_add_function(function, force=True)
        return graphs[""][1]

    def _make_subgraph_from_functions(self, graph: OnnxGraph):
        users: Dict[str, List[NodeProto]] = defaultdict(list)
        iter_order: Dict[str, OnnxGraph] = {}  # ordered hashset
        # graph.functions may not topologically sorted
        bfs: Dict[str, None] = OrderedDict()  # used as ordered set
        for n in graph:
            node: NodeProto = graph.nodes[n]["pb"]
            if node.op_type in graph.functions:
                users[node.op_type].append(node)
                iter_order[node.op_type] = graph
                bfs[node.op_type] = None
        while bfs:
            top_key = next(iter(bfs))
            f = graph.functions[top_key]
            del bfs[top_key]
            for node in f.node:
                if node.op_type in graph.functions:
                    users[node.op_type].append(node)
                    bfs[node.op_type] = None
                    iter_order[node.op_type] = extract_function(
                        graph,
                        f.node,
                        [(i, j) for i, j in zip(users[f.name][0].input, f.input)],
                        [(i, j) for i, j in zip(users[f.name][0].output, f.output)],
                    )
        extended_value_info: List[ValueInfoProto] = []
        for name, parent in iter_order.items():
            if len(users[name]) != 1:
                # Currently only function for a single node is OK to optimize
                # Otherwise we can't figure out how to do a general optimization
                # across different nodes.
                continue
            node = users[name][0]
            f = graph.functions[name]
            # pylint: disable=protected-access
            parent._value_info.extend(extended_value_info)
            model = extract_function(
                parent,
                f.node,
                [(i, j) for i, j in zip(node.input, f.input)],
                [(i, j) for i, j in zip(node.output, f.output)],
            )
            for i in chain(model.input, model.output):
                extended_value_info.append(make_value_info(i.name, i.type))
            yield name, (f.domain, model)

    def _make_function_from_subgraph(self, name: str, domain: str, model: OnnxGraph):
        # pylint: disable=protected-access
        model._keep_value_info = True
        h = model.model
        function = make_function(
            domain,
            name,
            list(model.inputs),
            list(model.outputs),
            h.graph.node,
            opset_imports=h.opset_import,
            value_info=h.graph.value_info,
        )
        return function

    @classmethod
    def print_all(cls):
        """Print the name of all passes."""
        print(PASSES, flush=True)

    @classmethod
    def print_l1(cls):
        """Print the name of all L1 passes."""
        print(L1, flush=True)

    @classmethod
    def print_l2(cls):
        """Print the name of all L2 passes."""
        print(L2, flush=True)

    @classmethod
    def print_l3(cls):
        """Print the name of all L3 passes."""
        print(L3, flush=True)

    @classmethod
    def print(cls, names: str | List[str]):
        """Print a specific pass or a set of passes."""
        print(PASSES.child(names), flush=True)

    def __repr__(self) -> str:
        return tabulate(
            [[i.__NAME__, i] for i in self.activated], ["PASS", "Func"], "grid"
        )


def print_pass_simple(pm: PassManager):
    """Print activated passes of a PassManager in a simple format."""

    msg = ""
    for n, i in enumerate(pm.activated):
        deps = []
        post = []
        # pylint: disable=protected-access
        for j in pm._expand_deps(i.__DEPS__):
            if j in PASSES:
                deps.append(j)
        for j in pm._expand_patches(i.__PATCHES__):
            if j in PASSES:
                post.append(j)
        msg += f"\n{n:<2} "
        if deps:
            msg += "[" + colored(",".join(deps), "yellow") + "] "
        msg += i.__NAME__
        if post:
            msg += " [" + colored(",".join(post), "magenta") + "]"
    if msg:
        print(f"Activated passes ([deps] pass [patches]):{msg}")
