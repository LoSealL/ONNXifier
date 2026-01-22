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
from onnx.helper import make_node, tensor_dtype_to_np_dtype
from onnx.onnx_pb import NodeProto

from .. import OnnxGraph, make_constant
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="clamp")
class Clamp(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/clamp-1.html

    https://onnx.ai/onnx/operators/onnx__Clip.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        min_value = float(self.get_attribute(ori_node, "min"))  # type: ignore
        max_value = float(self.get_attribute(ori_node, "max"))  # type: ignore
        try:
            prec = graph.tensor_type(ori_node.output[0])
        except ValueError:
            prec = graph.tensor_type(ori_node.input[0])
        dtype = tensor_dtype_to_np_dtype(prec)
        min_cst = make_constant(
            f"{ori_node.name}/min", np.array(min_value, dtype=dtype)
        )
        max_cst = make_constant(
            f"{ori_node.name}/max", np.array(max_value, dtype=dtype)
        )
        ori_node.input.extend([min_cst.output[0], max_cst.output[0]])
        self += [min_cst, max_cst]
        return make_node(
            "Clip", inputs=ori_node.input, outputs=ori_node.output, name=ori_node.name
        )
