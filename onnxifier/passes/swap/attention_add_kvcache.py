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
from onnx.helper import make_tensor_type_proto, make_value_info

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register(name="attention_add_kvcache", deps=["infer_shape"])
class AttentionAddKVCacheRewriter(Rewriter):
    """Add KV cache inputs/outputs to Attention op for autoregressive decoding.

    This rewriter transforms an attention op without KV cache to one with KV cache:

    Before (opset 23/24):
        Attention(input, weight, bias) -> (output)

    After:
        Attention(input, weight, bias, past_key, past_value) ->
            (output, present_key, present_value)

    And adds past_key, past_value as graph inputs and
    present_key, present_value as graph outputs.

    Optionally sets sequence length as a dynamic axis for the past/present tensors.
    """

    def __init__(self, sequence_length: int | str | None = None):
        super().__init__(SingleNodePattern("Attention"))
        self.sequence_length = sequence_length

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto]):
        node = nodes[0]

        # Only apply to opset 23 or 24
        if graph.opset_version not in (23, 24):
            return

        # Check if the node already has past_key/past_value inputs
        # Attention: input, weight, bias (opt), past_key (opt), past_value (opt)
        # Without KV cache: 3 inputs (input, weight, bias)
        # With KV cache: 5 inputs (input, weight, bias, past_key, past_value)
        has_kv_cache = len(node.input) >= 5 and node.input[3] and node.input[4]
        if has_kv_cache:
            return

        # Check if the node already has present_key/present_value outputs
        # Without KV cache: 1 output (output)
        # With KV cache: 3 outputs (output, present_key, present_value)
        has_present = len(node.output) >= 3 and node.output[1] and node.output[2]
        if has_present:
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

        # Create value info for past_key and past_value inputs
        past_shape = self._make_past_shape(input_shape, past_seq_len)
        past_key_info = make_value_info(
            past_key_name, make_tensor_type_proto(input_dtype, past_shape)
        )
        past_value_info = make_value_info(
            past_value_name, make_tensor_type_proto(input_dtype, past_shape)
        )

        # Create value info for present_key and present_value outputs
        present_shape = self._make_present_shape(input_shape, present_seq_len)
        present_key_info = make_value_info(
            present_key_name, make_tensor_type_proto(input_dtype, present_shape)
        )
        present_value_info = make_value_info(
            present_value_name,
            make_tensor_type_proto(input_dtype, present_shape),
        )

        # Add past_key and past_value as graph inputs
        graph.input.append(past_key_info)
        graph.inputs[past_key_name] = len(graph.input) - 1
        graph.input.append(past_value_info)
        graph.inputs[past_value_name] = len(graph.input) - 1

        # Add present_key and present_value as graph outputs
        graph.output.append(present_key_info)
        graph.outputs[present_key_name] = len(graph.output) - 1
        graph.output.append(present_value_info)
        graph.outputs[present_value_name] = len(graph.output) - 1

        # Extend node inputs to include past_key and past_value
        node.input.append(past_key_name)
        node.input.append(past_value_name)

        # Extend node outputs to include present_key and present_value
        node.output.append(present_key_name)
        node.output.append(present_value_name)

        # Update graph edges
        graph._node_to_out[node.name] = node.output
        for output_name in node.output:
            graph._out_to_node[output_name] = node.name

        # Add value info for new outputs
        graph.set_value_info(present_key_name, present_shape, input_dtype)
        graph.set_value_info(present_value_name, present_shape, input_dtype)

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
