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

from onnx import NodeProto

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register(name="attention_add_kvcache", deps=["infer_shape"])
class AttentionAddKVCacheRewriter(Rewriter):
    """Add KV cache inputs/outputs to Attention op for autoregressive decoding.

    This rewriter transforms an attention op without KV cache to one with KV cache:

    Before (opset 23/24):
        Attention(Q, K, V) -> (Y)

    After:
        Attention(Q, K, V, "", past_key, past_value) ->
            (Y, present_key, present_value)

    And adds past_key, past_value as graph inputs and
    present_key, present_value as graph outputs.

    Optionally sets sequence length as a dynamic axis for the past/present tensors.
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Attention"))

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto]):
        node = nodes[0]

        def _has_value(items, index: int) -> bool:
            return len(items) > index and bool(items[index])

        # Only apply to opset 23 or higher
        if graph.opset_version < 23:
            return

        # ONNX Attention(23/24) inputs are positional:
        #   0:Q, 1:K, 2:V, 3:attn_mask(opt), 4:past_key(opt),
        #   5:past_value(opt), 6:nonpad_kv_seqlen(opt, opset 24)
        # KV cache requires both past_key and past_value.
        has_past_key = _has_value(node.input, 4)
        has_past_value = _has_value(node.input, 5)
        has_kv_cache = has_past_key and has_past_value
        if has_kv_cache:
            return

        # Incomplete cache state is invalid; avoid mutating such nodes.
        if has_past_key != has_past_value:
            return

        # Spec forbids using nonpad_kv_seqlen together with past/present KV cache.
        if _has_value(node.input, 6):
            return

        # Outputs are positional:
        #   0:Y, 1:present_key(opt), 2:present_value(opt), 3:qk_matmul_output(opt)
        has_present_key = _has_value(node.output, 1)
        has_present_value = _has_value(node.output, 2)
        has_present = has_present_key and has_present_value
        if has_present:
            return

        # Incomplete present state is invalid; avoid mutating such nodes.
        if has_present_key != has_present_value:
            return

        # If output #1 is already used (for example qk_matmul_output only), adding
        # present outputs would change output semantics by position.
        if len(node.output) > 1:
            return

        # Generate names for past/present key/value tensors
        past_key_name = f"{node.name}/past_key"
        past_value_name = f"{node.name}/past_value"
        present_key_name = f"{node.name}/present_key"
        present_value_name = f"{node.name}/present_value"

        # Get shape and dtype from input tensor for creating value info
        input_shape, input_dtype = graph.tensor_info(node.input[0])

        # Keep both cache sequence lengths dynamic for autoregressive decoding.
        # ONNX shape symbols cannot encode arithmetic relations, so use separate
        # symbolic dimensions for past and present sequence lengths.
        past_seq_len = "past_seq_len"
        present_seq_len = "present_seq_len"

        # Resolve shapes for past_key and past_value inputs
        past_shape = self._make_past_shape(input_shape, past_seq_len)

        # Resolve shapes for present_key and present_value outputs
        present_shape = self._make_present_shape(input_shape, present_seq_len)

        # Register tensor metadata first so set_input/set_output can resolve
        # dtype/shape.
        graph.set_value_info(past_key_name, past_shape, input_dtype)
        graph.set_value_info(past_value_name, past_shape, input_dtype)
        graph.set_value_info(present_key_name, present_shape, input_dtype)
        graph.set_value_info(present_value_name, present_shape, input_dtype)

        # Fill optional positions by spec: [Q, K, V, attn_mask, past_key, past_value].
        while len(node.input) < 6:
            node.input.append("")
        node.input[4] = past_key_name
        node.input[5] = past_value_name

        # Fill optional output positions by spec: [Y, present_key, present_value].
        while len(node.output) < 3:
            node.output.append("")
        node.output[1] = present_key_name
        node.output[2] = present_value_name

        # Expose new tensors through public graph I/O APIs.
        graph.set_input(node, past_key_name)
        graph.set_input(node, past_value_name)
        graph.set_output(node, present_key_name)
        graph.set_output(node, present_value_name)

    def _make_past_shape(
        self, input_shape: list[int | str] | None, past_seq_len: int | str
    ) -> list[int | str]:
        """Create shape for past_key and past_value tensors.

        Past tensors have shape: [batch_size, num_heads, past_seq_len, head_dim]
        """
        if input_shape is None:
            return ["batch_size", "num_heads", "past_seq_len", "head_dim"]
        # Input shape is typically:
        # [batch_size, kv_seq_len, num_heads * head_dim]
        # or [batch_size, num_heads, kv_seq_len, head_dim]
        # For past_key/past_value:
        # [batch_size, num_heads, past_seq_len, head_dim]
        if len(input_shape) == 4:
            return [input_shape[0], input_shape[1], past_seq_len, input_shape[3]]
        elif len(input_shape) == 3:
            return ["batch_size", "num_heads", past_seq_len, "head_dim"]
        else:
            return ["batch_size", "num_heads", past_seq_len, "head_dim"]

    def _make_present_shape(
        self, input_shape: list[int | str] | None, present_seq_len: int | str
    ) -> list[int | str]:
        """Create shape for present_key and present_value tensors.

        Present tensors have shape:
        [batch_size, num_heads, present_seq_len, head_dim]
        where present_seq_len = past_seq_len + kv_seq_len
        """
        if input_shape is None:
            return ["batch_size", "num_heads", "present_seq_len", "head_dim"]
        if len(input_shape) == 4:
            return [input_shape[0], input_shape[1], present_seq_len, input_shape[3]]
        elif len(input_shape) == 3:
            return ["batch_size", "num_heads", present_seq_len, "head_dim"]
        else:
            return ["batch_size", "num_heads", present_seq_len, "head_dim"]
