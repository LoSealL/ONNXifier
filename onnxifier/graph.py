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

import gc
import os
import traceback
import warnings
from copy import deepcopy
from io import IOBase
from itertools import chain
from pathlib import Path
from typing import IO, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

import networkx as nx
import onnx
from onnx import external_data_helper as edh
from onnx.helper import (
    make_graph,
    make_model,
    make_operatorsetid,
    make_tensor_sequence_value_info,
    make_tensor_type_proto,
    make_tensor_value_info,
    make_value_info,
)

from .logger import debug, error
from .utils import chdir


def _unique_opset(opset_import: Sequence[onnx.OperatorSetIdProto]):
    domain_version: Dict[str, int] = {}
    for opset in opset_import:
        domain_version[opset.domain] = max(
            domain_version.get(opset.domain, 1), opset.version
        )
    return [make_operatorsetid(k, v) for k, v in domain_version.items()]


class OnnxGraph(nx.DiGraph):
    """Create a DAG from onnx graph.

    Args:
        model (onnx.ModelProto): use onnx.load_model to get the proto.
        base_dir (str): If the model contains external data, specify the base directory
            to load external data that relative to. If not specified, the current
            working directory is used as the base directory.
    """

    def __init__(
        self,
        model: Optional[onnx.ModelProto] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        if model is None:
            super().__init__()
        else:
            assert isinstance(model, onnx.ModelProto)
            super().__init__(name=model.graph.name)
            self._model = model
            self._node_to_out: Dict[str, Sequence[str]] = {}
            self._out_to_node: Dict[str, str] = {}
            self._functions = {f.name: f for f in model.functions}
            try:
                graph = onnx.shape_inference.infer_shapes(model).graph
            except onnx.shape_inference.InferenceError as ex:
                warnings.warn(f"Inferring shape failed, value info is not set:\n {ex}")
                graph = model.graph
            # WA: infer_shapes could cause graph to be empty in some cases
            if len(graph.node) == 0:
                graph = model.graph
            self.inputs = {i.name: n for n, i in enumerate(graph.input)}
            self.outputs = {i.name: n for n, i in enumerate(graph.output)}
            # read-only value info
            self._value_info = graph.value_info
            # write-only value info
            self._value_info_update: List[onnx.ValueInfoProto] = []
            self._keep_value_info = False
            for node in graph.node:
                self._add_onnx_node_internal(node)
            self._build_edges()
            # OnnxGraph save external tensors to one file, and this variable keeps
            # the parent directory of the tensor file, which is used to specify
            # a correct working directory when loading external data into onnx model.
            self._external_base: Optional[str] = self._get_external_base(base_dir)

    def _get_external_base(self, base_dir: str | None) -> str | None:
        err_msg = ""
        for t in self._get_all_tensors():
            if edh.uses_external_data(t):
                info = edh.ExternalDataInfo(t)
                if base_dir is None:
                    base_dir = info.basepath
                elif info.basepath and base_dir != info.basepath:
                    raise ValueError("External data have different base path!")
                if not os.path.exists(os.path.join(base_dir, info.location)):
                    err_msg += f"External {t.name} @{info.location} not found!\n"
        if err_msg:
            raise FileNotFoundError(err_msg)
        if base_dir == "":
            # WA: actually `basepath` is not used in current onnx
            base_dir = Path.cwd().as_posix()
        return base_dir

    def _add_onnx_node_internal(self, node: onnx.NodeProto) -> None:
        if not node.HasField("name"):
            node.name = f"{node.op_type}/{uuid4().hex}"
        name = node.name
        has_input = any(i in self.inputs for i in node.input)
        has_output = any(i in self.outputs for i in node.output)
        # clear out-dated edges on this node
        if name in self:
            self.remove_node(name)
        self.add_node(name, pb=node, has_input=has_input, has_output=has_output)
        self._node_to_out[name] = node.output
        for o in node.output:
            # we need handle cases that node A replaces node B, so _out_to_node
            # should change from B to A. Previously this behavior is UB.
            if o in self._out_to_node and self._out_to_node[o] != name:
                debug(
                    f"{self._out_to_node[o]} shares output {o} with {name}, "
                    "make sure the former will be removed."
                )
            self._out_to_node[o] = name

    def _build_edges(self):
        for n in self.nodes:
            node = self.nodes[n]["pb"]
            for i in node.input:
                if upstream := self._out_to_node.get(i):
                    self.add_edge(upstream, n, edge=i)

    def copy(self, as_view=False, onnx_copy=False):
        """Create a deep copy of the OnnxGraph.

        This method overrides NetworkX's copy method to ensure all ONNX-specific
        attributes are properly copied.

        Args:
            as_view (bool): If True, return a view (not implemented for OnnxGraph).
                           Defaults to False.
            onnx_copy (bool): If True, perform a deep copy of the model.
                           Defaults to False.
        """
        if not onnx_copy:
            return super().copy(as_view=as_view)
        if as_view:
            raise NotImplementedError("View copying is not supported for OnnxGraph")

        # Create a deep copy of the model to ensure all ONNX data is preserved
        copied_model = deepcopy(self._model)

        # Create a new OnnxGraph from the copied model
        copied_graph = OnnxGraph(copied_model, base_dir=self._external_base)

        # Copy any additional attributes that might have been modified
        copied_graph._keep_value_info = getattr(self, "_keep_value_info", False)
        copied_graph._value_info_update = deepcopy(self._value_info_update)

        return copied_graph

    def add_onnx_node(self, node: onnx.NodeProto) -> None:
        """Insert a node and its edges."""
        assert isinstance(node, onnx.NodeProto)
        self._add_onnx_node_internal(node)
        self._build_edges()
        # remove connected edge from inputs
        for output_name in node.output:
            for i, graph_input in enumerate(self.input):
                if graph_input.name == output_name:
                    self.input.pop(i)
                    self.inputs.pop(output_name)

    def remove_onnx_node(self, node: onnx.NodeProto | str, no_replace=False) -> None:
        """Remove a node from the graph"""
        assert isinstance(node, (onnx.NodeProto, str))
        if isinstance(node, str):
            name = node
            node = self.nodes[name]["pb"]
        else:
            name = node.name
        if name not in self:
            raise ValueError(f"Node {name} is not existing in the graph!")
        assert isinstance(node, onnx.NodeProto)
        if no_replace:
            # dump node output info to inputs
            for output_name in node.output:
                if self._out_to_node.get(output_name) != name:
                    # node is replaced
                    continue
                shape, dtype = self.tensor_info(output_name)
                self.input.append(
                    make_value_info(output_name, make_tensor_type_proto(dtype, shape))
                )
                self.inputs[output_name] = len(self.input) - 1
        self._node_to_out.pop(name)
        for output_name in node.output:
            if self._out_to_node[output_name] == name:
                self._out_to_node.pop(output_name)
                # output is not claimed by added node, remove it from graph
                if output_name in self.outputs:
                    self.remove_output(output_name)
        self.remove_node(name)

    def onnx_predecessors(self, n: onnx.NodeProto | str) -> List[onnx.NodeProto]:
        """Returns a list of predecessor nodes of n."""
        assert isinstance(n, (onnx.NodeProto, str))
        preds = self.predecessors(n if isinstance(n, str) else n.name)
        return [self.nodes[p]["pb"] for p in preds]

    def onnx_successors(self, n: onnx.NodeProto | str) -> List[onnx.NodeProto]:
        """Returns a list of successors nodes of n."""
        assert isinstance(n, (onnx.NodeProto, str))
        succs = self.successors(n if isinstance(n, str) else n.name)
        return [self.nodes[s]["pb"] for s in succs]

    def onnx_siblings(self, n: onnx.NodeProto | str) -> List[onnx.NodeProto]:
        r"""Returns a list of sibling nodes of n.

        A sibling node is defined as a node that:
        1. shares the same parent of `n`
        2. shares at least one input of `n`

        Example:

              P
              |  (e0)
             / \
            A   B

            Say node A is the sibling of node B, and vice versa. The common edge
            is [e0].
        """
        assert isinstance(n, (onnx.NodeProto, str))
        parents_and_edges: Dict[onnx.NodeProto, str] = {}
        if isinstance(n, str):
            node_name = n
        else:
            node_name = n.name
        for parent in self.onnx_predecessors(n):
            edge_data = self.get_edge_data(parent.name, node_name)
            parents_and_edges[parent] = str(edge_data["edge"])

        def _child_and_edge(node: onnx.NodeProto, edge: str):
            for child in self.onnx_successors(node):
                if edge == self.get_edge_data(node.name, child.name).get("edge"):
                    yield child

        siblings: List[onnx.NodeProto] = []
        for parent, edge in parents_and_edges.items():
            for child in _child_and_edge(parent, edge):
                if child.name != node_name:
                    siblings.append(child)
        return siblings

    def onnx_subgraph(self, nodes: Iterable[onnx.NodeProto | str]) -> "OnnxGraph":
        """Create a sub onnx graph from nodes"""
        sub: nx.DiGraph = self.subgraph(  # type: ignore
            (n if isinstance(n, str) else n.name for n in nodes)
        )
        subonnx = make_graph([], f"subgraph of {self.name}", inputs=[], outputs=[])
        # inherite initializers and value infos
        initializer_map = {i.name: i for i in self.initializer}
        subonnx.value_info.extend(self._value_info)

        def _compute_input(node):
            check_list = set(initializer_map.keys())
            for pred in sub.predecessors(node.name):
                check_list.update(sub.nodes[pred]["pb"].output)
            for input_name in node.input:
                if input_name and input_name not in check_list:
                    yield input_name
                if input_name in initializer_map:
                    subonnx.initializer.append(initializer_map[input_name])
                    initializer_map.pop(input_name)  # avoid duplicated initializer

        graph_inputs = set()
        for i in nx.topological_sort(sub):
            node = self.nodes[i]["pb"]
            subonnx.node.append(node)
            for output_name in node.output:
                is_outside = any(  # has an edge connected outside subgraph
                    output_name in j.input
                    for j in self.onnx_successors(node)
                    if j.name not in sub
                )
                # PR#176: if node is the output of the graph, we should not cut
                # it and must keep this output in the subgraph
                is_outside |= output_name in self.outputs
                if sub.out_degree(i) == 0 or is_outside:
                    shape, dtype = self.tensor_info(output_name)
                    subonnx.output.append(
                        make_value_info(
                            output_name, make_tensor_type_proto(dtype, shape)
                        )
                    )
            node_inputs = list(_compute_input(node))
            if node.op_type != "Constant" and node_inputs:
                graph_inputs.update(node_inputs)
        sorted_graph_inputs = sorted(graph_inputs)
        for input_name in sorted_graph_inputs:
            shape, dtype = self.tensor_info(input_name)
            subonnx.input.append(
                make_value_info(input_name, make_tensor_type_proto(dtype, shape))
            )
        sub_model = make_model(
            subonnx,
            doc_string=self._model.doc_string,
            domain=self._model.domain,
            model_version=self._model.model_version,
            ir_version=self._model.ir_version,
            opset_imports=self._model.opset_import,
            producer_name=self._model.producer_name,
            producer_version=self._model.producer_version,
            functions=self.functions.values(),
        )
        return OnnxGraph(sub_model, base_dir=self.external_base)

    def onnx_add_function(self, func: onnx.FunctionProto, force: bool = False) -> None:
        """Add a function to the graph.

        Args:
            func (onnx.FunctionProto): an onnx function proto.
            force (bool, optional): whether to forcely add the function.
                Defaults to False.
        """
        if func.name not in self._functions or force:
            self._functions[func.name] = func
            self._model.opset_import.extend(func.opset_import)

    def tensor_info(self, name: str) -> Tuple[List[int | str] | None, int]:
        """Get shape and dtype of a tensor by its name."""
        shape: Optional[List[int | str]] = None
        dtype = onnx.TensorProto.UNDEFINED
        if name in self.inputs or name in self.outputs:
            value_infos = list(chain(self.input, self.output))
        else:
            value_infos = self._value_info
        for value_info in value_infos:
            if value_info.name == name:
                if value_info.type.HasField("tensor_type"):
                    tensor_type = value_info.type.tensor_type
                elif value_info.type.HasField("sequence_type"):
                    seq_type = value_info.type.sequence_type.elem_type
                    # for now, sequence type must be a tensor type
                    if not seq_type.HasField("tensor_type"):
                        raise TypeError(f"Unsupported sequence type: {seq_type}")
                    tensor_type = seq_type.tensor_type
                else:
                    raise TypeError(f"Unsupported value type: {value_info.type}")
                dtype = tensor_type.elem_type
                if not tensor_type.HasField("shape"):
                    break  # no shape info, skip
                dims = [i for i in tensor_type.shape.dim]
                shape = [0 for _ in dims]
                for i, s in enumerate(dims):
                    if s.dim_param and s.dim_value == 0:
                        shape[i] = s.dim_param
                    else:
                        shape[i] = int(s.dim_value)
                break
        if shape is None:
            for tensor in self.initializer:
                if tensor.name == name:
                    shape = [int(i) for i in tensor.dims]
                    dtype = tensor.data_type
        return shape, dtype

    def tensor_shape(self, name: str) -> List[int | str]:
        """Get shape from a given tensor name."""
        shape, _ = self.tensor_info(name)
        if shape is None:
            raise ValueError(f"Can't find tensor shape with name '{name}'")
        return shape

    def static_tensor_shape(self, name: str) -> List[int]:
        """Get shape from a given tensor name, and ensure it is static."""
        shape = self.tensor_shape(name)
        static_shape: List[int] = []
        for ind, i in enumerate(shape):
            if isinstance(i, str) or i < 1:
                raise ValueError(f"shape[{ind}] is dynamic: {i}")
            static_shape.append(i)
        return static_shape

    def tensor_type(self, name: str) -> int:
        """Get tensor type from a given tensor name."""
        _, elem_type = self.tensor_info(name)
        if elem_type == onnx.TensorProto.UNDEFINED:
            raise ValueError(f"Can't find tensor dtype with name '{name}'")
        return elem_type

    def set_input(self, node: onnx.NodeProto | str, input_name: str):
        """Set the `input_name` of `node` as the graph input."""
        if isinstance(node, str):
            node = self.nodes[node]["pb"]
        assert isinstance(node, onnx.NodeProto)
        if input_name not in node.input:
            raise ValueError(f"Can't find {input_name} in {node.name}")
        if input_name in self.inputs:
            return  # already input
        shape, dtype = self.tensor_info(input_name)
        self.inputs[input_name] = len(self.input)
        if shape is None or dtype is None:
            raise ValueError(f"Can't find tensor shape and dtype of {input_name}")
        self.input.append(
            make_value_info(input_name, make_tensor_type_proto(dtype, shape))
        )

    def set_output(self, node: onnx.NodeProto | str, output_name: str):
        """Set the `output_name` of `node` as the graph output."""
        if isinstance(node, str):
            node = self.nodes[node]["pb"]
        assert isinstance(node, onnx.NodeProto)
        if output_name not in node.output:
            raise ValueError(f"Can't find {output_name} in {node.name}")
        replace = output_name in self.outputs
        shape, dtype = self.tensor_info(output_name)
        if replace:
            self.remove_output(output_name)
        self.outputs[output_name] = len(self.output)
        self._out_to_node[output_name] = node.name
        if shape is None or dtype is None:
            raise ValueError(f"Can't find tensor shape and dtype of {output_name}")
        self.output.append(
            make_value_info(output_name, make_tensor_type_proto(dtype, shape))
        )

    def remove_input(self, input_name: str):
        """Remove the `input_name` from the graph input."""
        ind = self.inputs.pop(input_name)
        self.input.pop(ind)
        for i in self.inputs:
            if self.inputs[i] > ind:
                self.inputs[i] -= 1

    def remove_output(self, output_name: str):
        """Remove the `output_name` from the graph output."""
        ind = self.outputs.pop(output_name)
        self.output.pop(ind)
        for i in self.outputs:
            if self.outputs[i] > ind:
                self.outputs[i] -= 1

    def rename_input(self, old_name: str, new_name: str):
        """Rename the `old_name` input to `new_name`."""
        self.inputs[new_name] = self.inputs.pop(old_name)
        self.input[self.inputs[new_name]].name = new_name
        for n in nx.topological_sort(self):
            if self.in_degree(n) > 0:  # type: ignore
                break
            input_node = self.nodes[n]["pb"]
            for ind, i in enumerate(input_node.input):
                if i == old_name:
                    input_node.input[ind] = new_name

    def rename_output(self, old_name: str, new_name: str):
        """Rename the `old_name` output to `new_name`."""
        self.outputs[new_name] = self.outputs.pop(old_name)
        self.output[self.outputs[new_name]].name = new_name
        self._out_to_node[new_name] = self._out_to_node.pop(old_name)
        output_node_name = self._out_to_node[new_name]
        output_node = self.nodes[output_node_name]["pb"]
        for ind, i in enumerate(output_node.output):
            if i == old_name:
                output_node.output[ind] = new_name
        self._node_to_out[output_node.name] = output_node.output

    def set_value_info(
        self, name: str, shape: Sequence[int | str], dtype: Optional[int] = None
    ):
        """Overwrite the value info of a tensor.

        Args:
            name (str): the name of the tensor.
            shape (List[int | str]): the shape of the tensor.
            dtype (int, optional): the data type of the tensor. Defaults to None.
        """
        for i, info in enumerate(self._value_info):
            if info.name == name:
                if dtype is None:
                    dtype = info.type.tensor_type.elem_type
                self._value_info.pop(i)
                self._value_info_update.append(
                    make_tensor_value_info(name, dtype, shape)
                )
                return
        if dtype is None:
            raise ValueError(
                f"dtype is required because value is not existing for {name}"
            )
        self._value_info_update.append(make_tensor_value_info(name, dtype, shape))

    def set_seqeuence_info(
        self, name: str, shape: Sequence[int | str], dtype: Optional[int] = None
    ):
        """Overwrite the value info of a sequence.

        Args:
            name (str): the name of the sequence.
            shape (List[int | str]): the shape of the tensor in the sequence.
            dtype (int, optional): the data type of the tensor. Defaults to None.
        """
        for i, info in enumerate(self._value_info):
            if info.name == name:
                if dtype is None:
                    dtype = info.type.tensor_type.elem_type
                self._value_info.pop(i)
                self._value_info_update.append(
                    make_tensor_sequence_value_info(name, dtype, shape)
                )
                return
        if dtype is None:
            raise ValueError(
                f"dtype is required because value is not existing for {name}"
            )
        self._value_info_update.append(
            make_tensor_sequence_value_info(name, dtype, shape)
        )

    @property
    def input(self):
        """Return a list of graph inputs value info."""
        return self._model.graph.input

    @property
    def output(self):
        """Return a list of graph outputs value info."""
        return self._model.graph.output

    @property
    def initializer(self):
        """Return a list of graph initializer tensors."""
        return self._model.graph.initializer

    @property
    def initializers(self) -> Dict[str, onnx.TensorProto]:
        """Return a dict of graph initializer tensors."""
        return {i.name: i for i in self.initializer}

    @property
    def model_version(self) -> int:
        """Return the model version of the model."""
        return self._model.model_version

    @property
    def ir_version(self) -> int:
        """Return the ir version of the model."""
        return self._model.ir_version

    @property
    def opset_version(self) -> int:
        """Return the opset version of the model."""
        return max(
            i.version for i in self._model.opset_import if i.domain in ("", "ai.onnx")
        )

    @opset_version.setter
    def opset_version(self, version: int):
        """Set the opset version of the model."""
        # keep only one ai.onnx opset
        for opset in list(self._model.opset_import):
            if opset.domain in ("", "ai.onnx"):
                self._model.opset_import.remove(opset)
        self._model.opset_import.append(make_operatorsetid("", version))

    @property
    def functions(self) -> Dict[str, onnx.FunctionProto]:
        """Return the functions inside the model."""
        return self._functions

    @property
    def external_base(self) -> Optional[str]:
        """Return the base directory of external data."""
        return self._external_base

    @external_base.setter
    def external_base(self, base_dir: str | os.PathLike | None):
        """Set the base directory of external data."""
        if base_dir:
            self._external_base = self._get_external_base(str(base_dir))
        else:
            self._external_base = self._get_external_base(None)

    @property
    def external_data(self) -> tuple[Path, ...]:
        """Yield all external data file locations."""

        if not self._external_base:
            return ()
        data = set()
        for t in self._get_all_tensors():
            if edh.uses_external_data(t):
                info = edh.ExternalDataInfo(t)
                location = Path(self._external_base) / info.location
                data.add(location)
        return tuple(data)

    def _get_all_tensors(self) -> Iterable[onnx.TensorProto]:
        # pylint: disable=protected-access
        yield from self.initializer
        for func in self.functions.values():
            yield from edh._get_attribute_tensors_from_graph(func)
        for name in self:
            node: onnx.NodeProto = self.nodes[name]["pb"]
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    yield attribute.t
                yield from attribute.tensors
                yield from edh._recursive_attribute_processor(
                    attribute, edh._get_initializer_tensors_from_graph
                )
                yield from edh._recursive_attribute_processor(
                    attribute, edh._get_attribute_tensors_from_graph
                )

    def convert_tensors_to_external(
        self,
        location: str | os.PathLike | Path,
        size_threshold: int = 1024,
    ):
        """Set all tensors with raw data as external data.

        Args:
            location (str | os.PathLike | Path): specify the external file to save to.
            size_threshold (int): Threshold for size of data. Only when
                tensor's data is >= the size_threshold it will be converted to external
                data. To convert every tensor with raw data to external data set
                size_threshold=0. Defaults to 1024.
        """
        location = Path(location)
        if not self._external_base:
            self._external_base = Path.cwd().as_posix()
            if location.is_absolute():
                self._external_base = location.parent.as_posix()
        if location.is_absolute() and location.is_relative_to(self._external_base):
            location = location.relative_to(self._external_base)
        elif location.is_absolute():
            raise ValueError(f"absolute location {location} must be relative to cwd.")
        # recreate the file
        with open(self._external_base / location, "wb"):
            pass
        for tensor in self._get_all_tensors():
            if tensor.HasField("raw_data") and len(tensor.raw_data) >= size_threshold:
                edh.set_external_data(tensor, location.as_posix())
        for tensor in self._get_all_tensors():
            if edh.uses_external_data(tensor) and tensor.HasField("raw_data"):
                edh.save_external_data(tensor, self._external_base)
                tensor.ClearField("raw_data")
        # free memory
        gc.collect()

    def restore_tensors_from_external(self):
        """Restore all external tensors into graph."""
        if not self._external_base:
            return

        for tensor in self._get_all_tensors():
            if edh.uses_external_data(tensor):
                info = edh.ExternalDataInfo(tensor)
                location = Path(self._external_base) / info.location
                with open(location, "rb") as data_file:
                    if info.offset:
                        data_file.seek(info.offset)

                    if info.length:
                        tensor.raw_data = data_file.read(info.length)
                    else:
                        tensor.raw_data = data_file.read()
                # After loading raw_data from external_data, change the state of tensors
                tensor.data_location = onnx.TensorProto.DEFAULT
                # and remove external data
                del tensor.external_data[:]
        self._external_base = None

    def canonicalize_external_tensors(self):
        """When `numpy_helper.to_array` is called, a tensor with external data
        is loaded into .raw_data filed. We need to remove this filed to pass
        the checker and also avoid saving extra data into onnx model.
        """
        for tensor in self._get_all_tensors():
            if edh.uses_external_data(tensor) and tensor.HasField("raw_data"):
                tensor.ClearField("raw_data")

    @property
    def model(self) -> onnx.ModelProto:
        """Make new model"""
        graph = deepcopy(self._model.graph)
        graph.ClearField("node")
        if self._keep_value_info:
            # remove duplicated values, later value_info will overwrite earlier ones
            values_map = {
                i.name: i for i in chain(self._value_info, self._value_info_update)
            }
            self._value_info_update = list(values_map.values())
        graph.ClearField("value_info")
        for n in nx.topological_sort(self):
            graph.node.append(self.nodes[n]["pb"])
        graph.value_info.extend(self._value_info_update)
        model = make_model(
            graph,
            doc_string=self._model.doc_string,
            domain=self._model.domain,
            model_version=self._model.model_version,
            ir_version=self._model.ir_version,
            opset_imports=_unique_opset(self._model.opset_import),
            producer_name=self._model.producer_name,
            producer_version=self._model.producer_version,
            functions=self._functions.values(),
        )
        model.metadata_props.extend(self._model.metadata_props)
        return model

    # pylint: disable=redefined-builtin
    def save(
        self,
        model_path: str | os.PathLike | IO[bytes],
        format: Optional[str] = None,
        save_as_external_data: bool = False,
        infer_shapes: bool = True,
        check: bool = True,
    ):
        """Serialize the graph to onnx model and save to model_path."""
        if save_as_external_data:
            if isinstance(model_path, (IO, IOBase)):
                location = (
                    self.name.replace(":", "_")
                    .replace("/", "_")
                    .replace("?", "")
                    .replace("!", "")
                    .replace(" ", "")
                )
            else:
                location = Path(model_path).resolve().with_suffix("")
            self.restore_tensors_from_external()
            self.convert_tensors_to_external(location=location)
        else:
            self.canonicalize_external_tensors()
        if infer_shapes:
            # infer shape using infer_shape pass
            # pylint: disable=import-outside-toplevel
            from .passes.globals.infer_shape import infer_shape

            graph_with_shape = infer_shape(self)
            # pylint: disable=protected-access
            graph_with_shape._keep_value_info = True
            model = graph_with_shape.model
        else:
            model = self.model
        if not isinstance(model_path, (str, os.PathLike)):
            format = "protobuf"
        elif format == "protobuf" or format is None:
            model_path = Path(model_path).with_suffix(".onnx")
        elif format == "textproto":
            model_path = Path(model_path).with_suffix(".pbtxt")
        elif format == "json":
            model_path = Path(model_path).with_suffix(".json")
        elif format == "onnxtxt":
            model_path = Path(model_path).with_suffix(".onnxtxt")
        if check:
            with chdir(self.external_base):
                try:
                    onnx.checker.check_model(model, full_check=True)
                except onnx.checker.ValidationError as ex:
                    error(f"onnx check failed (use -vv to show tracebacks):\n{ex}\n")
                    debug("\n".join(traceback.format_exception(ex)))
        onnx.save_model(model, model_path, format=format)
        return model_path
