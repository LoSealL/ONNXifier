"""
Copyright (C) 2026 The ONNXIFIER Authors.

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

from onnx import NodeProto, TensorProto

from ...graph import OnnxGraph
from ...logger import debug
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register(name="attention_add_mask", deps=["infer_shape"])
class AttentionAddMaskRewriter(Rewriter):
    """Add attention mask to Attention op for autoregressive decoding.

    This rewriter transforms an attention op without an attention mask to one with an
    attention mask:

    Before (opset 23/24):
        Attention(Q, K, V, "", ...) -> (Y, ...)

    After:
        Attention(Q, K, V, mask, ...) -> (Y, ...)

    And adds past_key, past_value as graph inputs and
    present_key, present_value as graph outputs.

    Optionally sets sequence length as a dynamic axis for the past/present tensors.
    """

    def __init__(
        self,
        mask_name: str = "full_attention_mask",
        total_sequence_length: int = 1,
        boolean_type: int = TensorProto.INT32,
    ):
        super().__init__(SingleNodePattern("Attention"))
        self.mask_name = mask_name
        self.total_sequence_length = total_sequence_length
        self.boolean_type = boolean_type

    def rewrite(
        self,
        graph: OnnxGraph,
        nodes: list[NodeProto],
        mask_name: str | None = None,
        total_sequence_length: int | None = None,
    ):
        node = nodes[0]

        def _has_value(items, index: int) -> bool:
            return len(items) > index and bool(items[index])

        # Only apply to opset 23 or higher
        if graph.opset_version < 23:
            return

        has_mask = _has_value(node.input, 3)
        if has_mask:
            return

        mask_name = mask_name or self.mask_name
        if mask_name in graph.inputs:
            node.input[3] = mask_name
            return

        # Get shape and dtype from input tensor for creating value info
        input_q_shape = graph.tensor_shape(node.input[0])
        q_seq_len = input_q_shape[-2]
        total_seq_len = total_sequence_length or self.total_sequence_length
        if _has_value(node.output, 1):
            output_kv_shape = graph.tensor_info(node.output[1])[0]
            if output_kv_shape is not None:
                total_seq_len = output_kv_shape[-2]

        node.input[3] = self.mask_name
        graph.set_value_info(
            self.mask_name, [q_seq_len, total_seq_len], self.boolean_type
        )
        graph.set_input(node, self.mask_name)
        debug("set graph input %s[%s]", mask_name, f"{q_seq_len}, {total_seq_len}")
