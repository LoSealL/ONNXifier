"""
Copyright (C) 2025 The ONNXIFIER Authors.

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

# pylint: disable=arguments-differ

from typing import List

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph, logger
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("transpose_softmax_to_last_dim")
class TransposeSoftmaxToLastDimRewriter(Rewriter):
    """XNNC doesn't support Softmax with dim other than -1, we need to transpose the
    input tensor and output tensor to change the dim of Softmax to -1.

    Before:

        a->softmax(dim=1)->b

    After:

        a->transpose()->softmax(dim=-1)->transpose()->b
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Softmax"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        axis = self.get_attribute(node, "axis")
        assert axis is None or isinstance(axis, (int, float, str))
        input_rank = len(graph.tensor_shape(node.input[0]))

        if axis is None or int(axis) in (-1, input_rank - 1):
            return

        axis = int(axis)
        if axis < 0:
            axis += input_rank
        perm = list(range(input_rank))
        perm[-1], perm[axis] = perm[axis], perm[-1]
        logger.debug(f"insert transpose before and after softmax: perm={perm}")

        trans_beg = make_node(
            "Transpose",
            inputs=[node.input[0]],
            outputs=[f"{node.name}/Transpose/beg_output0"],
            name=f"{node.name}/Transpose/beg",
            perm=perm,
        )
        trans_end = make_node(
            "Transpose",
            inputs=[f"{node.name}/Transpose/end_input0"],
            outputs=[node.output[0]],
            name=f"{node.name}/Transpose/end",
            perm=perm,
        )
        node.input[0] = trans_beg.output[0]
        node.output[0] = trans_end.input[0]
        self += [trans_beg, trans_end]
        self.set_attribute(node, "axis", -1)
