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

from typing import List

from onnx.onnx_pb import NodeProto, TensorProto

from . import IR_PASSES, OnnxGraph, Rewriter, SingleNodePattern


@IR_PASSES.register("eliminate_useless_floor")
class EliminateUselessFloorRewriter(Rewriter):
    """OpenVINO has a bug that floor on integers, it should be removed from the graph.

    Before:

        x(i64) -> Floor -> y(i64)

    After:

        x(i64) -> (Identity) -> y(i64)
    """

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Floor"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        _, input_type = graph.tensor_info(node.input[0])
        if input_type is None:
            return
        elif input_type in (
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.DOUBLE,
            TensorProto.BFLOAT16,
        ):
            return

        for next_n in self.get_output_node(node, 0):
            ind = 0
            for ind, input_name in enumerate(next_n.input):
                if input_name == node.output[0]:
                    break
            next_n.input[ind] = node.input[0]
        self -= node
