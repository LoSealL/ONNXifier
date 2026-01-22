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

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from .. import OnnxGraph
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="gather")
class Gather(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/gather-8.html

    https://onnx.ai/onnx/operators/onnx__Gather.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        axis = self.get_value(ori_node.input[2])
        if axis is None:
            raise RuntimeError(
                f"Do not support Gather ({ori_node.name}) with dynamic axis"
            )
        ori_node.input.pop(2)
        return make_node(
            "Gather",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            axis=int(axis),
        )
