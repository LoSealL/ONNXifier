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
from onnx.helper import make_node, np_dtype_to_tensor_dtype
from onnx.onnx_pb import NodeProto, TensorProto

from ...ir.mapping import ETYPE2DTYPE
from .. import OnnxGraph, cast_in
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="range")
class Range(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/generation/range-4.html

    https://onnx.ai/onnx/operators/onnx__Range.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        output_type = self.get_attribute(ori_node, "output_type")
        if output_type is not None:
            assert isinstance(output_type, str)
            assert output_type in ETYPE2DTYPE, f"Unsupported output_type: {output_type}"
            output_type = ETYPE2DTYPE[output_type]
            output_dtype = np_dtype_to_tensor_dtype(np.dtype(output_type))
        else:
            output_dtype = graph.tensor_type(ori_node.input[0])
        if output_dtype == TensorProto.UNDEFINED:
            output_dtype = TensorProto.INT64
        dtype0 = graph.tensor_type(ori_node.input[0])
        if dtype0 != output_dtype:
            self += cast_in(ori_node, 0, output_dtype)
        dtype1 = graph.tensor_type(ori_node.input[1])
        if dtype1 != output_dtype:
            self += cast_in(ori_node, 1, output_dtype)
        dtype2 = graph.tensor_type(ori_node.input[2])
        if dtype2 != output_dtype:
            self += cast_in(ori_node, 2, output_dtype)
        return make_node(
            "Range",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
        )
