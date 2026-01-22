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
from onnx.onnx_pb import NodeProto

from ...... import ONNXIFIER_OPSET
from .. import OnnxGraph, Rewriter, cast_in
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="prelu")
class PReLU(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/prelu-1.html

    https://onnx.ai/onnx/operators/onnx__PRelu.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        input_type = graph.tensor_type(ori_node.input[0])
        slope_type = graph.tensor_type(ori_node.input[1])
        if input_type != slope_type:
            self += cast_in(ori_node, 1, input_type)
        return make_node(
            "PRelu", inputs=ori_node.input, outputs=ori_node.output, name=ori_node.name
        )


@OP_CONVERT.register(name="relu")
class ReLU(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/relu-1.html

    https://onnx.ai/onnx/operators/onnx__Relu.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        return make_node(
            "Relu", inputs=ori_node.input, outputs=ori_node.output, name=ori_node.name
        )


@OP_CONVERT.register(name="gelu")
class Gelu(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/gelu-7.html

    https://onnx.ai/onnx/operators/onnx__Gelu.html
    """

    def __init__(self):
        super().__init__()
        self.register_post_hook(self.post_hook)

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        approximate = self.get_attribute(ori_node, "approximation_mode")
        assert isinstance(approximate, str)
        approximate = approximate.lower()
        return make_node(
            "Gelu",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            approximate="tanh" if approximate == "tanh" else "none",
        )

    def post_hook(self, graph: OnnxGraph):
        """Convert Gelu to subgraph for opset below 20."""
        # pylint: disable=import-outside-toplevel
        from onnxifier.passes.version_converter.opset19.gelu import Gelu as hook

        if ONNXIFIER_OPSET.version < 20:
            rewriter: Rewriter = hook()  # type: ignore
            return rewriter(graph)
        return graph


@OP_CONVERT.register(name="selu")
class Selu(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/activation/selu-1.html

    https://onnx.ai/onnx/operators/onnx__Selu.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        alpha = float(self.get_value_or_die(ori_node.input[1]).squeeze())
        gamma = float(self.get_value_or_die(ori_node.input[2]).squeeze())
        return make_node(
            "Selu",
            inputs=[ori_node.input[0]],
            outputs=ori_node.output,
            name=ori_node.name,
            alpha=alpha,
            gamma=gamma,
        )
