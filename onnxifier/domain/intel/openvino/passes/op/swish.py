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


@OP_CONVERT.register(name="swish")
class Swish(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/swish-4.html

    Swish(x) = x * sigmoid(x * beta)
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        sigmoid = make_node(
            "Sigmoid",
            inputs=[ori_node.input[0]],
            outputs=[f"{ori_node.name}/Sigmoid_output0"],
            name=f"{ori_node.name}/Sigmoid",
        )
        self += sigmoid
        if len(ori_node.output) > 1:
            # read beta
            beta_mul = make_node(
                "Mul",
                inputs=[ori_node.input[0], ori_node.input[1]],
                outputs=[f"{ori_node.name}/beta"],
                name=f"{ori_node.name}/beta",
            )
            sigmoid.input[0] = beta_mul.output[0]
            self += beta_mul

        return make_node(
            "Mul",
            inputs=[ori_node.input[0], sigmoid.output[0]],
            outputs=ori_node.output,
            name=ori_node.name,
        )
