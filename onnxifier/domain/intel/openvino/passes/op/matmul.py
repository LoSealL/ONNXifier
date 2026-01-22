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

from ...utils import text_to_boolean
from .. import OnnxGraph
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="matmul")
class MatMul(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/matrix/matmul-1.html

    https://onnx.ai/onnx/operators/onnx__MatMul.html
    """

    def _add_transpose(self, graph: OnnxGraph, node: NodeProto, index: int):
        shape = graph.tensor_shape(node.input[index])
        ndim = len(shape)
        perm = list(range(ndim))
        perm[-2:] = ndim - 1, ndim - 2
        trans_node = make_node(
            "Transpose",
            inputs=[node.input[index]],
            outputs=[f"{node.name}/transpose_{index}"],
            name=f"{node.name}/transpose_{index}",
            perm=perm,
        )
        node.input[index] = f"{node.name}/transpose_{index}"
        self += trans_node

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        transpose_a = self.get_attribute(ori_node, "transpose_a")
        assert isinstance(transpose_a, str)
        transpose_a = text_to_boolean(transpose_a)
        transpose_b = self.get_attribute(ori_node, "transpose_b")
        assert isinstance(transpose_b, str)
        transpose_b = text_to_boolean(transpose_b)
        if transpose_a:
            self._add_transpose(graph, ori_node, 0)
        if transpose_b:
            self._add_transpose(graph, ori_node, 1)
        return make_node(
            "MatMul",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
        )
