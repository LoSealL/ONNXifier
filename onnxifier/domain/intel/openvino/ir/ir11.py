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

from os import PathLike
from pathlib import Path
from typing import List, Optional, Union
from xml.etree import ElementTree

import networkx as nx
import numpy as np
import onnx
from onnx import numpy_helper
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_value_info,
    np_dtype_to_tensor_dtype,
)

from ..... import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, logger
from ... import IR_DOMAIN
from .mapping import ETYPE2DTYPE, PREC2DTYPE


def _node_to_dict(_node):
    return {i[0]: i[1] for i in _node.items()}


def _add_layer(graph, layer):
    node = layer.get("id")
    graph.add_node(
        node,
        name=layer.get("name"),
        type=layer.get("type"),
        version=layer.get("version"),
    )
    if (data := layer.find("data")) is not None:
        attrs = _node_to_dict(data)
        graph.add_node(node, **attrs)
    if (inputs := layer.find("input")) is not None:
        input_list = {}
        for port in inputs.iterfind("port"):
            port_id = port.get("id")
            input_list[port_id] = _node_to_dict(port)
            input_list[port_id].update(dim=[i.text for i in port.iterfind("dim")])
            if (attr := port.find("rt_info/attribute")) is not None:
                input_list[port_id].update(attr=_node_to_dict(attr))
        graph.add_node(node, inputs=input_list)
    if (outputs := layer.find("output")) is not None:
        output_list = {}
        for port in outputs.iterfind("port"):
            port_id = port.get("id")
            output_list[port_id] = _node_to_dict(port)
            output_list[port_id].update(dim=[i.text for i in port.iterfind("dim")])
            if (attr := port.find("rt_info/attribute")) is not None:
                output_list[port_id].update(attr=_node_to_dict(attr))
            if "name" not in output_list[port_id]:
                output_list[port_id].update(name=f"output{port_id}")
        graph.add_node(node, outputs=output_list)


def _add_edge(graph: nx.DiGraph, edge):
    beg = edge.get("from-layer")
    end = edge.get("to-layer")
    graph.add_edge(beg, end, src=edge.get("from-port"), dst=edge.get("to-port"))


def _load_const(graph, model_bin):
    for node in graph:
        try:
            if graph.nodes[node]["type"] != "Const":
                continue
            const_node = graph.nodes[node]
            size = int(const_node["size"])
            offset = int(const_node["offset"])
            dtype = ETYPE2DTYPE[const_node["element_type"]]
            bfloat16 = dtype == "bfloat16"
            if bfloat16:
                # need to convert bf16 to float16
                dtype = "float16"
            if const_node["shape"]:
                shape = list(map(int, const_node["shape"].split(",")))
            else:
                shape = []
            if np.prod(shape) == 0:
                data = np.empty([], dtype=dtype)
            else:
                if model_bin is not None:
                    raw = np.fromfile(
                        model_bin, dtype="uint8", count=size, offset=offset
                    )
                    if bfloat16:
                        # pylint: disable=import-outside-toplevel
                        import torch  # noqa: F401

                        tdata = torch.frombuffer(raw, dtype=torch.bfloat16)
                        data = tdata.to(torch.float16).numpy()
                    data = np.frombuffer(raw.tobytes(), dtype=dtype).reshape(shape)
                else:
                    data = np.zeros(shape, dtype=dtype)
            const_node["data"] = data
        except Exception:
            logger.error(f"exceptions on node:{node} {graph.nodes[node]['name']}")
            raise


def _set_input_and_output(graph):
    inputs = []
    outputs = []
    for node in graph:
        if graph.nodes[node]["type"] == "Parameter":
            inputs.append(node)
        elif graph.nodes[node]["type"] == "Result":
            outputs.append(node)
    graph.graph["input"] = inputs
    graph.graph["output"] = outputs


def _graph_to_onnx(graph: nx.MultiDiGraph) -> onnx.ModelProto:
    onnx_nodes = []
    onnx_inputs = []
    onnx_outputs = []
    values_info = []
    for node_id in nx.topological_sort(graph):
        node_attrs = graph.nodes[node_id].copy()
        node_name = node_attrs.pop("name")
        node_type = node_attrs.pop("type")
        node_inputs = node_attrs.pop("inputs", {})
        node_outputs = node_attrs.pop("outputs", {})
        outputs_name: List[str] = []
        outputs_shape: List[List[int]] = []
        outputs_dtype: List[str] = []
        for _, i in sorted(node_outputs.items(), key=lambda x: int(x[0])):
            outputs_name.append(f"{node_name}_{i['name']}")
            outputs_shape.append(list(map(int, i["dim"])))
            outputs_dtype.append(PREC2DTYPE[i.get("precision", "FP32")])
        input_maps = {}
        for pred_node in graph.predecessors(node_id):
            pred_attrs = graph.nodes[pred_node]
            edges = graph.get_edge_data(pred_node, node_id)
            for edge in edges.values():
                pred_output = pred_attrs["outputs"][edge["src"]]
                # should sort by port id
                input_maps[int(edge["dst"])] = (
                    f"{pred_attrs['name']}_{pred_output['name']}"
                )
        input_names = [i[1] for i in sorted(input_maps.items(), key=lambda x: x[0])]
        for name, shape, typestr in zip(outputs_name, outputs_shape, outputs_dtype):
            if typestr == "bfloat16":
                dtype = int(onnx.TensorProto.BFLOAT16)
            else:
                dtype = np_dtype_to_tensor_dtype(np.dtype(typestr))
            values_info.append(make_tensor_value_info(name, dtype, shape))
        try:
            if node_type == "Parameter":
                par_shape: List[int | str] = []
                for i, dim in enumerate(node_attrs.pop("shape").split(",")):
                    if dim == "?":
                        par_shape.append(f"D{i}")
                    elif dim != "":
                        par_shape.append(int(dim))
                typestr = ETYPE2DTYPE[node_attrs.pop("element_type")]
                if typestr == "bfloat16":
                    dtype = int(onnx.TensorProto.BFLOAT16)
                else:
                    dtype = np_dtype_to_tensor_dtype(np.dtype(typestr))
                onnx_inputs.append(make_tensor_value_info(node_name, dtype, par_shape))
                input_names = [node_name]
            elif node_type == "Result":
                result = next(iter(node_inputs.values()))
                res_shape: List[int | str] = list(map(int, result["dim"]))
                for i, dim in enumerate(res_shape):
                    if dim == -1:
                        res_shape[i] = f"Y{i}"
                if "precision" not in result:
                    logger.warning(f"precision not defined in {node_name}(Result)")
                    dtype = int(onnx.TensorProto.UNDEFINED)
                else:
                    typestr = PREC2DTYPE[result["precision"]]
                    if typestr == "bfloat16":
                        dtype = int(onnx.TensorProto.BFLOAT16)
                    else:
                        dtype = np_dtype_to_tensor_dtype(np.dtype(typestr))
                onnx_outputs.append(make_tensor_value_info(node_name, dtype, res_shape))
                outputs_name = [node_name]
            elif node_type == "Const":
                node_attrs.pop("offset")
                node_attrs.pop("size")
                node_attrs.pop("element_type")
                data = node_attrs.pop("data")
                node_attrs["value"] = numpy_helper.from_array(data)
            onnx_nodes.append(
                make_node(
                    node_type,
                    inputs=input_names,
                    outputs=outputs_name,
                    domain=IR_DOMAIN.domain,
                    name=node_name,
                    **node_attrs,
                )
            )
        except Exception:
            logger.error(f"Can't make node {node_name}({node_type})")
            raise

    onnx_graph = make_graph(
        onnx_nodes,
        name=graph.name,
        inputs=onnx_inputs,
        outputs=onnx_outputs,
        value_info=values_info,
    )
    return make_model(
        onnx_graph,
        producer_name="ONNXIFIER",
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[IR_DOMAIN, ONNXIFIER_OPSET],
    )


def ir_to_onnx(
    model_path: Union[str, PathLike],
    model_bin: Optional[Union[str, PathLike]] = None,
) -> onnx.ModelProto:
    """Parse OpenVINO IR format XML to ``onnx.ModelProto``.

    Args:
        model_path (Union[str, PathLike]): A URL to model xml file.
        model_bin (Optional[Union[str, PathLike]], optional): A URL to model bin file.
            If not specified, search at the same directory of xml file with same name.

    Raises:
        NotImplementedError: If IR version it not supported.

    Returns:
        onnx.ModelProto: An ONNX model.
    """
    etree = ElementTree.parse(model_path)
    name = etree.getroot().get("name")
    if (ir_ver := etree.getroot().get("version")) not in ("10", "11"):
        raise NotImplementedError(
            f"IRv{ir_ver} is not supported, expected IRv10 or IRv11"
        )

    if model_bin is None and Path(model_path).with_suffix(".bin").exists():
        model_bin = Path(model_path).with_suffix(".bin")
    graph_info = {}
    if (rt_info := etree.find("rt_info")) is not None:
        graph_info = {
            i.tag: i.get("value")
            for i in filter(lambda x: "value" in x.keys(), rt_info.iter())
        }
    if "name" in graph_info:
        graph_info.pop("name")
    graph = nx.MultiDiGraph(name=name, **graph_info)  # type: ignore
    for layer in etree.iterfind("layers/layer"):
        _add_layer(graph, layer)
    # load zeros if model_bin is not existed
    _load_const(graph, model_bin)
    if model_bin is None:
        logger.warning("model_bin is not existed, use all zeros for weights")
    for edge in etree.iterfind("edges/edge"):
        _add_edge(graph, edge)
    _set_input_and_output(graph)
    return _graph_to_onnx(graph)
