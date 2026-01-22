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

import numpy as np
from onnx.helper import make_node
from onnx.numpy_helper import from_array
from onnx.onnx_pb import NodeProto

from ...utils import text_to_integers
from .. import OnnxGraph
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="constant")
class Const(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/constant-1.html

    https://onnx.ai/onnx/operators/onnx__Constant.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        value = self.get_attribute(ori_node, "value")
        shape = self.get_attribute(ori_node, "shape")
        assert isinstance(value, np.ndarray)
        if shape is not None:
            assert isinstance(shape, str)
            shape = [i for i in text_to_integers(shape) if i != 0]
            value = value.reshape(shape)
        return make_node(
            "Constant",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            value=from_array(value),
        )
