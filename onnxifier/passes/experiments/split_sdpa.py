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

# pylint: disable=arguments-differ
from typing import List, Literal, Optional

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("split_sdpa")
class SplitSDPARewriter(Rewriter):
    """Split scaled dot-product attention (SDPA) from hidden dimension like
    FlashAttention.

    Before:

        H = softmax(Q @ K) @ V

    After:

        H = Sum(exp(Q @ K_i) @ V_i) / Sum(exp(Q @ K_i))

    Note:
        The dimensions should be static.
    """

    def __init__(self):
        pattern = GraphPattern()
        act = SingleNodePattern()
        pattern.add_edge(SingleNodePattern("MatMul"), act)
        pattern.add_edge(act, SingleNodePattern("MatMul"))
        super().__init__(pattern)

    def rewrite(
        self,
        graph: OnnxGraph,
        nodes: List[NodeProto],
        blocks: int = 64,
        mode: Optional[Literal["split_by_hidden_dims", "split_by_heads"]] = None,
    ):
        if mode == "split_by_hidden_dims":
            self.split_by_hidden_dims(graph, nodes, blocks)
        elif mode == "split_by_heads":
            self.split_by_heads(graph, nodes, 1)
        else:
            self.split_by_heads(graph, nodes, 1)

    def split_by_hidden_dims(
        self, graph: OnnxGraph, nodes: List[NodeProto], blocks: int = 64
    ):
        mm0, act, mm1 = nodes
        # query = self.get_input_node_or_die(mm0, 0)
        key = self.get_input_node_or_die(mm0, 1)
        value = self.get_input_node_or_die(mm1, 1)

        hidden_dims = graph.static_tensor_shape(key.output[0])[-1]
        if hidden_dims % blocks != 0:
            raise ValueError(
                f"Hidden dimension {hidden_dims} is not divisible by blocks {blocks}"
            )

        split0_output = [
            f"{mm0.name}/split_output{i}" for i in range(hidden_dims // blocks)
        ]
        split0 = make_node(
            "Split",
            inputs=[key.output[0]],
            outputs=split0_output,
            name=f"{mm0.name}/split",
            num_outputs=hidden_dims // blocks,
            axis=-1,
        )
        split1_output = [
            f"{mm1.name}/split_output{i}" for i in range(hidden_dims // blocks)
        ]
        split1 = make_node(
            "Split",
            inputs=[value.output[0]],
            outputs=split1_output,
            name=f"{mm1.name}/split",
            num_outputs=hidden_dims // blocks,
            axis=-2,
        )
        self += [split0, split1]
        sum_inputs = []
        for i, (key_out, value_out) in enumerate(zip(split0_output, split1_output)):
            sub_mm0 = make_node(
                "MatMul",
                inputs=[mm0.input[0], key_out],
                outputs=[f"{mm0.name}_{i}_output0"],
                name=f"{mm0.name}_{i}",
            )
            sub_act = make_node(
                act.op_type,
                inputs=[sub_mm0.output[0]],
                outputs=[f"{act.name}_{i}_output0"],
                name=f"{act.name}_{i}",
            )
            sub_act.attribute.extend(act.attribute)
            sub_mm1 = make_node(
                "MatMul",
                inputs=[sub_act.output[0], value_out],
                outputs=[f"{mm1.name}_{i}_output0"],
                name=f"{mm1.name}_{i}",
            )
            sum_inputs.append(sub_mm1.output[0])
            self += [sub_mm0, sub_act, sub_mm1]

        reduce_sum = make_node(
            "Sum", inputs=sum_inputs, outputs=[mm1.output[0]], name=f"{mm1.name}/sum"
        )

        self += reduce_sum
        self -= [mm0, mm1, act]

    def split_by_heads(
        self, graph: OnnxGraph, nodes: List[NodeProto], blocks: int = 64
    ):
        mm0, act, mm1 = nodes
        query = self.get_input_node_or_die(mm0, 0)
        key = self.get_input_node_or_die(mm0, 1)
        value = self.get_input_node_or_die(mm1, 1)

        heads = graph.static_tensor_shape(query.output[0])[1]
        if heads % blocks != 0:
            raise ValueError(f"Heads {heads} is not divisible by blocks {blocks}")

        split0a_output = [
            f"{mm0.name}/split0a_output{i}" for i in range(heads // blocks)
        ]
        split0a = make_node(
            "Split",
            inputs=[query.output[0]],
            outputs=split0a_output,
            name=f"{mm0.name}/split0a",
            num_outputs=heads // blocks,
            axis=1,
        )
        split0b_output = [
            f"{mm0.name}/split0b_output{i}" for i in range(heads // blocks)
        ]
        split0b = make_node(
            "Split",
            inputs=[key.output[0]],
            outputs=split0b_output,
            name=f"{mm0.name}/split0b",
            num_outputs=heads // blocks,
            axis=1,
        )
        split1_output = [f"{mm1.name}/split_output{i}" for i in range(heads // blocks)]
        split1 = make_node(
            "Split",
            inputs=[value.output[0]],
            outputs=split1_output,
            name=f"{mm1.name}/split",
            num_outputs=heads // blocks,
            axis=1,
        )
        self += [split0a, split0b, split1]
        sum_inputs = []
        for i, (query_out, key_out, value_out) in enumerate(
            zip(split0a_output, split0b_output, split1_output)
        ):
            sub_mm0 = make_node(
                "MatMul",
                inputs=[query_out, key_out],
                outputs=[f"{mm0.name}_{i}_output0"],
                name=f"{mm0.name}_{i}",
            )
            sub_act = make_node(
                act.op_type,
                inputs=[sub_mm0.output[0]],
                outputs=[f"{act.name}_{i}_output0"],
                name=f"{act.name}_{i}",
            )
            sub_act.attribute.extend(act.attribute)
            sub_mm1 = make_node(
                "MatMul",
                inputs=[sub_act.output[0], value_out],
                outputs=[f"{mm1.name}_{i}_output0"],
                name=f"{mm1.name}_{i}",
            )
            sum_inputs.append(sub_mm1.output[0])
            self += [sub_mm0, sub_act, sub_mm1]

        concat = make_node(
            "Concat",
            inputs=sum_inputs,
            outputs=[mm1.output[0]],
            axis=1,
            name=f"{mm1.name}/concat",
        )

        self += concat
        self -= [mm0, mm1, act]
