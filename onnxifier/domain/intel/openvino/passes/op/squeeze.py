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

import numpy as np
import onnx
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from .. import OnnxGraph, cast_in
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="squeeze")
class Squeeze(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/shape/squeeze-1.html

    https://onnx.ai/onnx/operators/onnx__Squeeze.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        if len(ori_node.input) > 1:
            axes_type = graph.tensor_type(ori_node.input[1])
            axes_shape = graph.tensor_shape(ori_node.input[1])
            numerical_shape = list(filter(lambda x: isinstance(x, int), axes_shape))
            if np.prod(numerical_shape) == 0:  # type: ignore
                ori_node.input.pop(1)
            elif axes_type != onnx.TensorProto.INT64:
                # add a cast
                self += cast_in(ori_node, 1, onnx.TensorProto.INT64)
        return make_node(
            ori_node.op_type,
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
        )
