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
import numpy as np
from onnx import NodeProto
from onnx.helper import make_tensor_type_proto, make_value_info

from ... import OnnxGraph, logger
from ...domain.trt.ops.vit_attention_plugin import (
    from_onnx_attention,
    vit_attention_plugin_schema,
)
from .. import PASSES
from ..utils import make_constant
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

    Args:
        bake_cu_seqlens (bool): Whether to bake cu_seqlens as a constant input.
            If input shapes are static and known, baking cu_seqlens as [0, s0, s0+s1,
            ..., s0+...+sN]. Defaults to True.
    """

    def rewrite(
        self,
        graph: OnnxGraph,
        nodes: list[NodeProto],
        bake_cu_seqlens: bool = True,
        **_,
    ):
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
        if not bake_cu_seqlens or not self._bake_seqlens(graph, plugin_op):
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

    def _bake_seqlens(self, graph: OnnxGraph, node: NodeProto) -> bool:
        """Bake cu_seqlens as a constant input if possible."""
        # For now ViT attention is only for Qwen-VL series, so seq_len is
        # the 1st dimension of graph input.
        input_shape = graph.tensor_shape(graph.input[0].name)
        if len(input_shape) == 2:
            seq_len = input_shape[0]
        elif len(input_shape) == 3:
            seq_len = input_shape[1]
        else:
            logger.warning("Unexpected graph input shape, can't bake cu_seqlens.")
            return False
        if not isinstance(seq_len, int) or seq_len <= 0:
            logger.warning("Symbolic seq_len, can't bake cu_seqlens.")
            return False
        cu_seqlens = make_constant("cu_seqlens", np.array([0, seq_len], dtype=np.int32))
        node.input[3] = cu_seqlens.output[0]
        self += cu_seqlens
        seqlen_carrier = make_constant(
            "max_seqlen_carrier", np.array([seq_len], dtype=np.int32)
        )
        node.input[4] = seqlen_carrier.output[0]
        self += seqlen_carrier
        return True

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
