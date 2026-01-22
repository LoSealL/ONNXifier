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


@OP_CONVERT.register(name="normalize_l2")
class NormalizeL2(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/normalization/normalize-l2-1.html

    https://onnx.ai/onnx/operators/onnx__LpNormalization.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        axes = self.get_value(ori_node.input[1])
        if axes is not None and axes.size == 1:
            # static single axis can turn into LpNormalization
            axis = int(axes.squeeze())
            return make_node(
                "LpNormalization",
                inputs=[ori_node.input[0]],
                outputs=ori_node.output,
                name=ori_node.name,
                axis=axis,
                p=2,
            )

        # use reduce_l2 and div
        reduce_l2 = make_node(
            "ReduceL2",
            inputs=ori_node.input,
            outputs=[f"{ori_node.name}/ReduceL2_output0"],
            name=f"{ori_node.name}/ReduceL2",
        )
        self += reduce_l2
        return make_node(
            "Div",
            inputs=[ori_node.input[0], reduce_l2.output[0]],
            outputs=ori_node.output,
            name=ori_node.name,
        )
