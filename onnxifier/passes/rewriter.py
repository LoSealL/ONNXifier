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

from abc import ABCMeta, abstractmethod
from collections.abc import Generator, Sequence, Sized
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

import numpy as np
import onnx
from onnx import FunctionProto, numpy_helper
from onnx.helper import make_attribute, make_function, make_operatorsetid

from ..graph import OnnxGraph
from ..logger import error
from .pattern import Pattern
from .utils import attribute_value, evaluate_on_node


class RewriterRepeat(Enum):
    """Repeat mode of the rewriter. By default, the rewriter will only match the graph
    once. If the rewriter still wants to match the rewritten graph recursively, set to
    ``INFINITE``.
    """

    ONCE = 1
    INFINITE = 9999  # maximum repeat times


class MetaRewriter(ABCMeta):
    """Override the attribute of new rewriter class."""

    def __new__(mcs, name, bases, namespace, /, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        setattr(cls, "__NAME__", name)
        setattr(cls, "__DEPS__", [])
        setattr(cls, "__PATCHES__", [])
        return cls


class Rewriter(metaclass=MetaRewriter):
    """OnnxGraph rewriter to modify the graph.

    Args:
        pattern (Pattern): a pattern to match nodes
        repeat (RewriterRepeat): repeat mode of the rewriter. Default is ONCE.
    """

    __NAME__: str = "Rewriter"
    __DEPS__: List[str] = []
    __PATCHES__: List[str] = []

    def __init__(self, pattern: Pattern, repeat=RewriterRepeat.ONCE):
        assert isinstance(pattern, Pattern)
        self.pattern = pattern
        repeat_value = repeat.value if isinstance(repeat, RewriterRepeat) else repeat
        self.repeat = min(int(repeat_value), RewriterRepeat.INFINITE.value)
        self.node_to_add: Set[onnx.NodeProto] = set()
        self.node_to_remove: Set[onnx.NodeProto] = set()
        self.pre_hooks: Dict[int, Callable[[OnnxGraph], OnnxGraph]] = {}
        self.post_hooks: Dict[int, Callable[[OnnxGraph], OnnxGraph]] = {}
        # record how many patterns have been matched and rewritten
        self.num_rewrites = 0

    @abstractmethod
    def rewrite(self, graph: OnnxGraph, nodes: List[onnx.NodeProto], *args, **kwargs):
        """Implement how to rewrite matched nodes in the graph

        Args:
            graph (OnnxGraph): an onnx graph
            nodes (List[NodeProto]): a list of matched nodes
        """

    def add(self, nodes: Sequence[onnx.NodeProto] | onnx.NodeProto):
        """Append nodes to be added to the graph

        Args:
            nodes: a single node or a list of nodes to add
        """
        if isinstance(nodes, onnx.NodeProto):
            self.node_to_add.add(nodes)
        elif isinstance(nodes, Sequence):
            assert all(isinstance(i, onnx.NodeProto) for i in nodes)
            self.node_to_add.update(nodes)
        else:
            raise TypeError(
                f"Expect to add a node or a list of nodes, but got {type(nodes)}."
            )

    def remove(self, nodes: Sequence[onnx.NodeProto] | onnx.NodeProto):
        """Remove nodes from the graph.

        Args:
            nodes: a single node or a list of nodes to remove
        """
        if isinstance(nodes, onnx.NodeProto):
            self.node_to_remove.add(nodes)
        elif isinstance(nodes, Sequence):
            assert all(isinstance(i, onnx.NodeProto) for i in nodes)
            self.node_to_remove.update(nodes)
        else:
            raise TypeError(
                f"Expect to add a node or a list of nodes, but got {type(nodes)}."
            )

    def match_and_rewrite(
        self,
        graph: OnnxGraph,
        *args,
        names: str | Sequence[str] | None = None,
        _specify_node_names: Set[str] | None = None,
        **kwargs,
    ) -> OnnxGraph:
        """Look up for matched patterns in the graph and rewrite it.

        Args:
            graph (OnnxGraph): onnx graph
            names (Sequence[str], optional): Only optimize nodes with these
                names. Defaults to None. A public interface for users.
            _specify_node_names (set[str], optional): Only optimize nodes with these
                names. Defaults to None. A private interface for PassManager.

        Returns:
            OnnxGraph: rewritten graph
        """
        repeat = self.repeat
        self.num_rewrites = 0
        for hook_fn in self.pre_hooks.values():
            graph = hook_fn(graph)
        _names: Set[str] = set()
        if names:
            _names.update([names] if isinstance(names, str) else names)
        if _specify_node_names:
            _names.update(_specify_node_names)
        while repeat > 0:
            matched_nodes = self.pattern.match(graph, _names)
            if matched_nodes is None:
                return graph  # no nothing
            if not isinstance(matched_nodes, Generator):
                raise RuntimeError(
                    f"{type(self.pattern)} match function is invalid! It should yield "
                    f"matched nodes, but it returns {type(matched_nodes)}"
                )
            # pylint: disable=attribute-defined-outside-init
            self._graph = graph
            i = None
            for i, nodes in enumerate(matched_nodes):
                if isinstance(nodes, onnx.NodeProto):
                    nodes = [nodes]
                if self.node_to_remove:
                    # filter nodes that been removed in previous pass
                    nodes = list(filter(lambda n: n not in self.node_to_remove, nodes))
                rewriter_can_run = len(nodes) != 0
                if isinstance(self.pattern, Sized):
                    # shared match been modified by previous rewriter
                    rewriter_can_run = len(nodes) == len(self.pattern)
                if rewriter_can_run:
                    try:
                        self.rewrite(graph, nodes, *args, **kwargs)
                    except Exception:
                        nodes_name = ",".join(n.name for n in nodes)
                        error(f"Rewrite nodes [{nodes_name}] failed.")
                        raise
                    finally:
                        self.num_rewrites += 1
            if i is None:
                break  # no match found
            for node in self.node_to_add:
                graph.add_onnx_node(node)
            added_node_names = set([n.name for n in self.node_to_add])
            for node in self.node_to_remove:
                if node.name in added_node_names:
                    # this node has been replaced by a new node, skip it
                    continue
                graph.remove_onnx_node(node)
            self.node_to_add.clear()
            self.node_to_remove.clear()
            repeat -= 1
        for hook_fn in self.post_hooks.values():
            graph = hook_fn(graph)
        return graph

    def get_input_node(
        self, node: onnx.NodeProto, i_or_s: int | str
    ) -> onnx.NodeProto | None:
        """Get the input node to the i-th input."""
        graph = self.graph
        # pylint: disable=protected-access
        if isinstance(i_or_s, int):
            i = i_or_s
            if i < 0:
                i += len(node.input)
            assert i < len(node.input), f"index {i} exceeds input number"
            if name := graph._out_to_node.get(node.input[i]):
                return graph.nodes[name]["pb"]
        else:
            name = i_or_s
            if name := graph._out_to_node.get(name):
                return graph.nodes[name]["pb"]

    def get_input_node_or_die(
        self, node: onnx.NodeProto, i_or_s: int | str
    ) -> onnx.NodeProto:
        """Get the input node to the i-th input or raise an exception if not found."""
        input_node = self.get_input_node(node, i_or_s)
        if input_node is None:
            raise ValueError(f"Failed to find input node for {node.name}({i_or_s})")
        return input_node

    def get_input_nodes(self, node: onnx.NodeProto) -> List[onnx.NodeProto | None]:
        """Get all input nodes for the given node."""
        return [self.get_input_node(node, i) for i in node.input]

    def get_output_node(
        self, node: onnx.NodeProto, i_or_s: int | str = 0
    ) -> List[onnx.NodeProto]:
        """Get the output node from the i-th output."""
        graph = self.graph
        if isinstance(i_or_s, int):
            i = i_or_s
            if i < 0:
                i += len(node.output)
            assert i < len(node.output), f"index {i} exceeds output number"
            port = node.output[i]
        else:
            port = i_or_s
        return [s for s in graph.onnx_successors(node) if port in s.input]

    def get_attribute(self, node: onnx.NodeProto, name: str, default=None):
        """Try to get the value of an attribute of the node.

        Args:
            node: a node
            name: name of the attribute
            default: set a default value if attribute is optional and not found
        """
        for attr in node.attribute:
            if attr.name == name:
                return attribute_value(attr)
        return default

    def set_attribute(self, node: onnx.NodeProto, name: str, value: Any):
        """Set a new value to an attribute of the node."""
        for attr in node.attribute:
            if attr.name == name:
                match attr.type:
                    case onnx.AttributeProto.FLOAT:
                        attr.f = float(value)
                    case onnx.AttributeProto.INT:
                        attr.i = int(value)
                    case onnx.AttributeProto.STRING:
                        attr.s = str(value).encode()
                    case onnx.AttributeProto.TENSOR:
                        attr.t.CopyFrom(numpy_helper.from_array(value))
                    case onnx.AttributeProto.TYPE_PROTO:
                        attr.tp.CopyFrom(value)
                    case onnx.AttributeProto.FLOATS:
                        attr.ClearField("floats")
                        attr.floats.extend(value)
                    case onnx.AttributeProto.INTS:
                        attr.ClearField("ints")
                        attr.ints.extend(value)
                    case onnx.AttributeProto.STRINGS:
                        attr.ClearField("strings")
                        attr.strings.extend(i.encode() for i in value)
                    case onnx.AttributeProto.TENSORS:
                        attr.ClearField("tensors")
                        attr.tensors.extend(value)
                    case onnx.AttributeProto.GRAPHS:
                        attr.ClearField("graphs")
                        attr.graphs.extend(value)
                    case onnx.AttributeProto.TYPE_PROTOS:
                        attr.ClearField("type_protos")
                        attr.type_protos.extend(value)
                    case _:
                        raise ValueError(f"Unsupported attribute type: {attr.type}")
                return
        node.attribute.append(make_attribute(name, value))

    def remove_attribute(self, node: onnx.NodeProto, name: str):
        """Remove an attribute from the node."""
        for attr in list(node.attribute):
            if attr.name == name:
                node.attribute.remove(attr)
                break

    def get_value(
        self, node: onnx.NodeProto | str, output_name: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Get value from a constant node."""
        external_base = self.graph.external_base or ""
        if isinstance(node, str):
            for init in self.graph.initializer:
                if init.name == node:
                    return numpy_helper.to_array(init, external_base)
            # value not found in initializer, try to find it from constant.
            # pylint: disable=protected-access
            cst_node_name = self.graph._out_to_node.get(node)
            if not cst_node_name:
                return
            return self.get_value(self.graph.nodes[cst_node_name]["pb"], node)
        elif node.op_type == "Constant":
            attr = node.attribute[0]
            if attr.name == "value":
                return numpy_helper.to_array(attr.t, external_base)
            if attr.name == "value_float":
                return np.array(attr.f, dtype=np.float32)
            if attr.name == "value_floats":
                return np.array(attr.floats, dtype=np.float32)
            if attr.name == "value_int":
                return np.array(attr.i, dtype=np.int64)
            if attr.name == "value_ints":
                return np.array(attr.ints, dtype=np.int64)
            if attr.name == "value_string":
                return np.array(attr.s)
            if attr.name == "value_strings":
                return np.array(attr.strings)
            raise TypeError(f"Unknown attribute type of Constant: {attr.name}")
        elif node.op_type == "Shape":
            return np.asarray(self.graph.tensor_shape(node.input[0]))
        # try to fold all upstreaming nodes
        return evaluate_on_node(self.graph, node, output_name)

    def get_value_or_die(
        self, node: onnx.NodeProto | str, output_name: Optional[str] = None
    ) -> np.ndarray:
        """Get value and raise exception if value is not evaluated."""
        value = self.get_value(node, output_name)
        if value is None:
            node_name = node if isinstance(node, str) else node.name
            raise ValueError(f"Failed to evaluate value for {node_name}({output_name})")
        return value

    def make_function_from_graph(
        self, graph: OnnxGraph, domain: str, type_name: str
    ) -> FunctionProto:
        """Make onnx local function from a subgraph."""
        h = graph.model.graph
        function = make_function(
            domain=domain,
            fname=type_name,
            inputs=list(graph.inputs),
            outputs=list(graph.outputs),
            nodes=h.node,
            opset_imports=[
                make_operatorsetid(domain, 1),
                make_operatorsetid("", graph.opset_version),
            ],
            value_info=h.value_info,
        )
        return function

    def register_post_hook(self, hook: Callable[[OnnxGraph], OnnxGraph]) -> int:
        """Register a post-rewrite hook function.

        A hook function must be a callable that takes an OnnxGraph as input and returns
        an OnnxGraph as output. It will be called after the last run of rewrite.

        Args:
            hook: a hook function to be registered

        Returns:
            int: a unique id of the hook function
        """
        if not callable(hook):
            raise TypeError(f"Expect a callable, but got {type(hook)}")
        uid = uuid4().node
        self.post_hooks[uid] = hook
        return uid

    def remove_post_hook(self, uid: int):
        """Remove a post-rewrite hook function.

        Args:
            uid: a unique id of the hook function to be removed
        """
        self.post_hooks.pop(uid)

    def register_pre_hook(self, hook: Callable[[OnnxGraph], OnnxGraph]) -> int:
        """Register a pre-rewrite hook function.

        A hook function must be a callable that takes an OnnxGraph as input and returns
        an OnnxGraph as output. It will be called before each run of rewrite.

        Args:
            hook: a hook function to be registered

        Returns:
            int: a unique id of the hook function
        """
        if not callable(hook):
            raise TypeError(f"Expect a callable, but got {type(hook)}")
        uid = uuid4().node
        self.pre_hooks[uid] = hook
        return uid

    def remove_pre_hook(self, uid: int):
        """Remove a pre-rewrite hook function.

        Args:
            uid: a unique id of the hook function to be removed
        """
        self.pre_hooks.pop(uid)

    @property
    def graph(self) -> OnnxGraph:
        """Get the current graph."""
        return self._graph

    def __call__(self, graph: OnnxGraph, *args, **kwargs) -> OnnxGraph:
        """Make rewriter a callable."""
        return self.match_and_rewrite(graph, *args, **kwargs)

    def __iadd__(self, nodes: Sequence[onnx.NodeProto] | onnx.NodeProto):
        self.add(nodes)
        return self

    def __isub__(self, nodes: Sequence[onnx.NodeProto] | onnx.NodeProto):
        self.remove(nodes)
        return self
