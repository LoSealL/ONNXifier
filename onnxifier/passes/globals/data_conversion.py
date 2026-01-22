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
import re

import onnx
from onnx import TensorProto
from onnx.helper import make_attribute
from onnx.numpy_helper import from_array, to_array

from ... import OnnxGraph, logger
from .. import L1, PASSES
from ..utils import cast_in


@L1.register()
def half_to_float(graph: OnnxGraph) -> OnnxGraph:
    """Convert half consts and values to float32."""

    for node in graph:
        node_pb = graph.nodes[node]["pb"]
        if node_pb.op_type == "Constant":
            tensor = node_pb.attribute[0].t
            if tensor.data_type == TensorProto.FLOAT16:
                array = to_array(tensor).astype("float32")
                attr = make_attribute(key="value", value=from_array(array))
                node_pb.attribute.pop()
                node_pb.attribute.append(attr)
        elif node_pb.op_type == "Cast":
            # if cast target is fp16, change it to fp32
            for attr in node_pb.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            array = to_array(init).astype("float32")
            init.data_type = TensorProto.FLOAT
            init.raw_data = from_array(array).raw_data
    for io in itertools.chain(graph.input, graph.output):
        if io.type.tensor_type.elem_type == TensorProto.FLOAT16:
            io.type.tensor_type.elem_type = TensorProto.FLOAT
    return graph


def canonicalize_resize(graph: OnnxGraph, op_name: str):
    """The scales input of Resize must be float32."""
    assert op_name in graph.nodes
    node_pb = graph.nodes[op_name]["pb"]
    scales = node_pb.input[2]
    if scales:
        graph.add_onnx_node(cast_in(node_pb, 2, TensorProto.FLOAT))


def dtype_canonicalize(graph: OnnxGraph, op_type: str, op_name: str):
    """Canonicalize data type for operators that don't support fp16."""
    if op_type == "Resize":
        canonicalize_resize(graph, op_name)
    # TODO: add more operators here


@PASSES.register()
def float_to_half(graph: OnnxGraph) -> OnnxGraph:
    """Convert float32 consts and values to half."""

    for node in graph:
        node_pb = graph.nodes[node]["pb"]
        if node_pb.op_type == "Constant":
            tensor = node_pb.attribute[0].t
            if tensor.data_type == TensorProto.FLOAT:
                array = to_array(tensor).astype("float16")
                attr = make_attribute(key="value", value=from_array(array))
                node_pb.attribute.pop()
                node_pb.attribute.append(attr)
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            array = to_array(init).astype("float16")
            init.data_type = TensorProto.FLOAT16
            init.raw_data = from_array(array).raw_data
    for io in itertools.chain(graph.input, graph.output):
        if io.type.tensor_type.elem_type == TensorProto.FLOAT:
            io.type.tensor_type.elem_type = TensorProto.FLOAT16
    # fix operators that don't support fp16
    try:
        onnx.checker.check_model(graph.model, full_check=True)
    except onnx.shape_inference.InferenceError as e:
        op_type = re.findall(r"op_type\:(\w+)", f"{e}")[0]
        op_name = re.findall(r"node name\: (.*)\)", f"{e}")[0]
        logger.debug(f"Operator {op_name}({op_type}) check failed, fixing it...")
        dtype_canonicalize(graph, op_type, op_name)
    return graph
