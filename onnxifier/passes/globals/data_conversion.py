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

# pylint: disable=arguments-differ

import itertools
from typing import List

from onnx import NodeProto, TensorProto
from onnx.helper import make_attribute
from onnx.numpy_helper import from_array, to_array

from ... import OnnxGraph
from .. import L1, PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
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


class CanonicalizeResizeScale(Rewriter):
    """The scales input of Resize must be float32."""

    def __init__(self):
        super().__init__(SingleNodePattern("Resize"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        if node.input[2]:
            self += cast_in(node, 2, TensorProto.FLOAT)


@PASSES.register(patch=[CanonicalizeResizeScale])
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
        elif node_pb.op_type == "Cast":
            # if cast target is fp32, change it to fp16
            for attr in node_pb.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            array = to_array(init).astype("float16")
            init.data_type = TensorProto.FLOAT16
            init.raw_data = from_array(array).raw_data
    for io in itertools.chain(graph.input, graph.output):
        if io.type.tensor_type.elem_type == TensorProto.FLOAT:
            io.type.tensor_type.elem_type = TensorProto.FLOAT16
    return graph
