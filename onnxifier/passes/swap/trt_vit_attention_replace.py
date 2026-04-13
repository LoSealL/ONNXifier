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

from ... import OnnxGraph, logger
from ...domain.trt.ops.vit_attention_plugin import (
    from_onnx_attention,
    vit_attention_plugin_schema,
)
from .. import PASSES
from .trt_attention_replace import (
    TRTAttentionRewriter,
    _elem_type_from_schema,
    _ensure_trt_opset,
)

_INPUT_NAMES = {
    "cu_seqlens": "cu_seqlens",
    "max_seqlen_carrier": "max_seqlen_carrier",
}

_INPUT_SHAPES = {
    "cu_seqlens": ["batch_size + 1"],
    "max_seqlen_carrier": ["max_seqlen"],
}


@PASSES.register(
    name="trt_vit_attention_replace", deps=["attention_fill_heads_and_dim"]
)
class TRTViTAttentionRewriter(TRTAttentionRewriter):
    """Replace ONNX Attention node with TensorRT ViTAttentionPlugin.

    This rewriter transforms a standard ONNX Attention (opset 24) node into
    a TensorRT ViTAttentionPlugin node, designed for Vision Transformer
    attention with head-major layout [total_S, H, D] for Q/K/V, no KV cache,
    and no RoPE.

    Before:
        Attention(q, k, v, attn_mask) -> (y)

    After:
        ViTAttentionPlugin(q, k, v, cu_seqlens, max_seqlen_carrier) -> (attn_output)

    Shared graph inputs are added: cu_seqlens, max_seqlen_carrier.
    """

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto], **_):
        node = nodes[0]
        orig_q_input = node.input[0]
        q_shape = graph.tensor_shape(node.input[0])
        q_num_heads = self.get_attribute(node, "q_num_heads", 0)
        head_size = self._infer_head_size(q_shape, q_num_heads)
        if not isinstance(head_size, int) or head_size <= 0:
            logger.debug(f"Can't infer head_size from input, skip node: {node.name}")
            return
        if not self._transpose_qkv_inputs(graph, node, head_size):
            logger.debug(f"Can't transpose qkv inputs, skip node: {node.name}")
            return

        # Build kwargs with actual input names
        plugin_kwargs = {
            "cu_seqlens": _INPUT_NAMES["cu_seqlens"],
            "max_seqlen_carrier": _INPUT_NAMES["max_seqlen_carrier"],
        }

        plugin_op = from_onnx_attention(node, head_size, **plugin_kwargs)
        _ensure_trt_opset(graph)
        self._add_shared_inputs(graph)
        self -= node
        self += plugin_op
        self._collect_plugin_value_info(
            graph,
            node,
            plugin_op,
            q_shape,
            q_num_heads,
            head_size,
            orig_q_input,
        )

    def _add_shared_inputs(self, graph: OnnxGraph):
        """Add shared inputs to the graph for all ViT attention nodes."""
        for key, name in _INPUT_NAMES.items():
            if name in graph.inputs:
                continue
            graph_input = make_value_info(
                name,
                make_tensor_type_proto(
                    _elem_type_from_schema(vit_attention_plugin_schema, key),
                    _INPUT_SHAPES[key],
                ),
            )
            graph.input.append(graph_input)
            graph.inputs[name] = len(graph.input) - 1
