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

from typing import List, Optional

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from onnxifier import logger

from .. import OnnxGraph
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="transpose")
class Transpose(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/transpose-1.html

    https://onnx.ai/onnx/operators/onnx__Transpose.html
    """

    def _guess_perm(
        self, src_shape: List[int | str], target_shape: List[int | str]
    ) -> Optional[List[int]]:
        assert len(src_shape) == len(target_shape)
        # do not modify the original list directly
        src_shape_copy = src_shape.copy()
        perm = []
        for i in target_shape:
            perm.append(src_shape_copy.index(i))
            src_shape_copy[perm[-1]] = -1
        return perm

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        perm = self.get_value(ori_node.input[1])
        guessed_perm = None
        src_shape, _ = graph.tensor_info(ori_node.input[0])
        target_shape, _ = graph.tensor_info(ori_node.output[0])
        if src_shape and target_shape:
            guessed_perm = self._guess_perm(src_shape, target_shape)
        if perm is None or all(p == 0 for p in perm):
            logger.warning(f"perm is empty, use the guessed perm={guessed_perm}")
            perm = guessed_perm
        if perm is None:
            raise RuntimeError(
                f"Do not support transpose ({ori_node.name}) with dynamic perm"
            )
        ori_node.input.pop(1)
        return make_node(
            "Transpose",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            perm=list(map(int, perm)),
        )
