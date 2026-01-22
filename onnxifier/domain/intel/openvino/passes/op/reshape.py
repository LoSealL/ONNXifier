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

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto, TensorProto

from .. import OnnxGraph, cast_in
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="reshape")
class Reshape(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/shape/reshape-1.html

    https://onnx.ai/onnx/operators/onnx__Reshape.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        shape_type = graph.tensor_type(ori_node.input[1])
        if shape_type != TensorProto.INT64:
            self += cast_in(ori_node, 1, TensorProto.INT64)
        return make_node(
            "Reshape",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
        )
