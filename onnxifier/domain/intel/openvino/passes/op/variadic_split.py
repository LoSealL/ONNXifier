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

import onnx
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from .. import OnnxGraph, cast_in, make_constant
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="variadic_split")
class VariadicSplit(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/variadic-split-1.html

    https://onnx.ai/onnx/operators/onnx__Split.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        axis = self.get_value_or_die(ori_node.input[1])
        ori_node.input.pop(1)
        dtype = graph.tensor_type(ori_node.input[-1])
        if dtype != onnx.TensorProto.INT64:
            self += cast_in(ori_node, -1, onnx.TensorProto.INT64)
        # split value must >= 0, but variadic_split can have negative value
        value = self.get_value(ori_node.input[-1])
        if value is not None:
            value = value.copy()
            rank = graph.tensor_shape(ori_node.input[0])[axis]
            if isinstance(rank, int):
                if (value < 0).sum() > 1:
                    raise ValueError(f"Only one negative value is allowed! Got {value}")
                value[value < 0] = rank - sum(value[value >= 0])
                pos_value = make_constant(f"{ori_node.name}/pos_split", value)
                ori_node.input[-1] = pos_value.output[0]
                self += pos_value
        return make_node(
            "Split",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            axis=int(axis),
        )
