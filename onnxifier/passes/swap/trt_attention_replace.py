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

import numpy as np
import onnx
from onnx import NodeProto
from onnx.helper import make_node, make_tensor_type_proto, make_value_info

from ... import OnnxGraph, logger
from ...domain.trt import TRT_IR_DOMAIN
from ...domain.trt.ops.attention_plugin import (
    attention_plugin_schema,
    from_onnx_attention,
)
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant

_INPUT_NAMES = {
    "context_lengths": "context_lengths",
    "rope_rotary_cos_sin": "rope_rotary_cos_sin",
    "kvcache_start_index": "kvcache_start_index",
}

_INPUT_SHAPES: dict[str, list[str | int]] = {
    "context_lengths": ["batch_size"],
    "rope_rotary_cos_sin": ["batch_size", "max_position_embeddings", "rotary_dim"],
    "kvcache_start_index": ["batch_size"],
}


def _elem_type_from_schema(schema: onnx.defs.OpSchema, input_name: str) -> int:
    """Infer ONNX tensor element type from an OpSchema input declaration."""
    for inp in schema.inputs:
        if inp.name != input_name:
            continue
        type_str = inp.type_str
        if not type_str.startswith("tensor(") or not type_str.endswith(")"):
            break
        elem_name = type_str[len("tensor(") : -1].upper()
        return onnx.TensorProto.DataType.Value(elem_name)
    raise ValueError(f"Cannot infer tensor element type for input '{input_name}'.")


def _ensure_trt_opset(graph: OnnxGraph):
    """Ensure TensorRT custom opset is declared in model opset imports."""
    for opset in graph._model.opset_import:  # pylint: disable=protected-access
        if (
            opset.domain == TRT_IR_DOMAIN.domain
            and opset.version >= TRT_IR_DOMAIN.version
        ):
            return
    graph._model.opset_import.append(TRT_IR_DOMAIN)  # pylint: disable=protected-access


@PASSES.register(name="trt_attention_replace", deps=["attention_fill_heads_and_dim"])
class TRTAttentionRewriter(Rewriter):
    """Replace ONNX Attention node with TensorRT AttentionPlugin.

    This rewriter transforms a standard ONNX Attention (opset 24) node into
    a TensorRT AttentionPlugin node with RoPE, KV cache, and attention.

    Before:
        Attention(q, k, v, attn_mask, past_key, past_value)
            -> (y, present_key, present_value)

    After:
        AttentionPlugin(q, k, v, past_key_value, context_lengths,
                        rope_rotary_cos_sin, kvcache_start_index,
                        attention_mask, attention_pos_id,
                        k_v_scale_quant_orig)
            -> (attn_output, present_key_value)

    Shared graph inputs are added: context_lengths, rope_rotary_cos_sin,
    kvcache_start_index.
    """

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Attention"))
        self._pending_value_infos: dict[str, tuple[list[int | str], int]] = {}
        self.register_pre_hook(self._reset_pending_value_info)
        self.register_post_hook(self._flush_pending_value_info)

    def rewrite(  # pylint: disable=arguments-differ
        self,
        graph: OnnxGraph,
        nodes: list[NodeProto],
        *,
        enable_tree_attention: int = 0,
        enable_fp8_kv_cache: int = 0,
        sliding_window_size: int = -1,
        attention_pos_id: str = "",
        k_v_scale_quant_orig: str = "",
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
        plugin_kwargs: dict = {
            "enable_tree_attention": enable_tree_attention,
            "enable_fp8_kv_cache": enable_fp8_kv_cache,
            "sliding_window_size": sliding_window_size,
            "attention_pos_id": attention_pos_id,
            "k_v_scale_quant_orig": k_v_scale_quant_orig,
            "context_lengths": _INPUT_NAMES["context_lengths"],
            "rope_rotary_cos_sin": _INPUT_NAMES["rope_rotary_cos_sin"],
            "kvcache_start_index": _INPUT_NAMES["kvcache_start_index"],
        }

        plugin_op = from_onnx_attention(node, head_size, **plugin_kwargs)
        _ensure_trt_opset(graph)
        self._add_shared_inputs(graph)
        self._add_kv_cache_io(graph, node, plugin_op)
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

    def _infer_head_size(self, q_shape: list[int | str], q_num_heads: int) -> int:
        """Infer head_size from ONNX Attention Q tensor shape."""
        if len(q_shape) == 4:
            # [batch, heads, seq, dim]
            last = q_shape[-1]
            return int(last) if isinstance(last, int) and last > 0 else 0
        if len(q_shape) == 3:
            # [batch, seq, heads*dim]
            hidden = q_shape[-1]
            if (
                isinstance(hidden, int)
                and hidden > 0
                and q_num_heads > 0
                and hidden % q_num_heads == 0
            ):
                return hidden // q_num_heads
        return 0

    def _transpose_qkv_inputs(
        self, graph: OnnxGraph, node: NodeProto, head_size: int
    ) -> bool:
        if len(node.input) < 3:
            return False

        q_heads = self.get_attribute(node, "q_num_heads", 0)
        kv_heads = self.get_attribute(node, "kv_num_heads", q_heads)
        if not isinstance(q_heads, int) or q_heads <= 0:
            return False
        if not isinstance(kv_heads, int) or kv_heads <= 0:
            return False

        converted_nodes: list[NodeProto] = []
        for idx, heads in ((0, q_heads), (1, kv_heads), (2, kv_heads)):
            input_name = node.input[idx]
            if not input_name:
                return False
            in_shape = graph.tensor_shape(input_name)

            base = f"{node.name}/trt_qkv_{idx}"

            if len(in_shape) == 3:
                # [batch, seq, heads*dim] -> [batch, seq, heads, dim]
                shape_name = f"{base}/shape_to_4d"
                shape_node = make_constant(
                    shape_name,
                    np.array(
                        [in_shape[0], in_shape[1], heads, head_size], dtype=np.int64
                    ),
                )
                converted_nodes.extend(
                    (
                        shape_node,
                        make_node(
                            "Reshape",
                            [input_name, shape_node.output[0]],
                            [f"{base}/reshape_to_4d_Output0"],
                            name=f"{base}/reshape_to_4d",
                        ),
                    )
                )
            elif len(in_shape) == 4:
                # [batch, heads, seq, dim] -> [batch, seq, heads, dim]
                perm = [0, 2, 1, 3]
                converted_nodes.append(
                    make_node(
                        "Transpose",
                        [input_name],
                        [f"{base}/transpose_Output0"],
                        name=f"{base}/transpose",
                        perm=perm,
                    )
                )
            else:
                return False

            # [batch, seq, heads, dim] -> [batch, seq, heads * dim]
            shape_name = f"{base}/shape_to_3d"
            shape_node = make_constant(
                shape_name,
                np.array([in_shape[0], -1, heads * head_size], dtype=np.int64),
            )
            converted_nodes.extend(
                (
                    shape_node,
                    make_node(
                        "Reshape",
                        [converted_nodes[-1].output[0], shape_node.output[0]],
                        [f"{base}/reshape_to_3d_Output0"],
                        name=f"{base}/reshape_to_3d",
                    ),
                )
            )
            node.input[idx] = converted_nodes[-1].output[0]

        if converted_nodes:
            self += converted_nodes
        return True

    def _add_shared_inputs(self, graph: OnnxGraph):
        """Add shared inputs to the graph for all attention nodes."""
        batch_size = 0  # infer batch size
        for output in graph.outputs:
            batch_size = graph.tensor_shape(output)[0]
            if isinstance(batch_size, str):
                batch_size = 0
            if batch_size > 0:
                break
        for key, name in _INPUT_NAMES.items():
            if name in graph.inputs:
                continue
            shape = _INPUT_SHAPES[key].copy()
            if shape[0] == "batch_size" and batch_size > 0:
                shape[0] = batch_size
            graph_input = make_value_info(
                name,
                make_tensor_type_proto(
                    _elem_type_from_schema(attention_plugin_schema, key),
                    shape,
                ),
            )
            graph.input.append(graph_input)
            graph.inputs[name] = len(graph.input) - 1

    def _add_kv_cache_io(self, graph: OnnxGraph, node: NodeProto, plugin_op: NodeProto):
        """Replace split KV cache graph IO with merged key/value graph IO."""
        # past_key/past_value -> past_key_value (plugin input[3])
        if len(node.input) > 5 and node.input[4] and node.input[5]:
            past_key = node.input[4]
            past_value = node.input[5]
            if past_key in graph.inputs and past_value in graph.inputs:
                merged_past_name = past_key.replace("key", "key_value")
                past_shape, past_dtype = graph.tensor_info(past_key)
                value_shape, _ = graph.tensor_info(past_value)
                merged_past_shape = self._stack_shape_axis1(past_shape, value_shape)
                if merged_past_name not in graph.inputs:
                    graph.input.append(
                        make_value_info(
                            merged_past_name,
                            make_tensor_type_proto(past_dtype, merged_past_shape),
                        )
                    )
                    graph.inputs[merged_past_name] = len(graph.input) - 1
                if past_key != merged_past_name:
                    graph.remove_input(past_key)
                graph.remove_input(past_value)
                plugin_op.input[3] = merged_past_name

        # present_key/present_value -> present_key_value (plugin output[1])
        if len(node.output) > 2 and node.output[1] and node.output[2]:
            present_key = node.output[1]
            present_value = node.output[2]
            if present_key in graph.outputs and present_value in graph.outputs:
                merged_present_name = present_key.replace("key", "key_value")
                present_shape, present_dtype = graph.tensor_info(present_key)
                value_shape, _ = graph.tensor_info(present_value)
                merged_present_shape = self._stack_shape_axis1(
                    present_shape, value_shape
                )
                if merged_present_name not in graph.outputs:
                    graph.output.append(
                        make_value_info(
                            merged_present_name,
                            make_tensor_type_proto(
                                present_dtype,
                                merged_present_shape,
                            ),
                        )
                    )
                    graph.outputs[merged_present_name] = len(graph.output) - 1
                if present_key != merged_present_name:
                    graph.remove_output(present_key)
                graph.remove_output(present_value)
                plugin_op.output[1] = merged_present_name

    def _stack_shape_axis1(
        self,
        lhs: list[int | str] | None,
        rhs: list[int | str] | None,
    ) -> list[int | str]:
        """Compute output shape for stack([lhs, rhs], axis=1)."""
        if lhs is None:
            lhs = ["batch_size", "heads", "seq_len", "head_size"]
        if rhs is None:
            rhs = lhs
        # for now both key/value shape must be same
        for l_dim, r_dim in zip(lhs, rhs):
            if isinstance(l_dim, int) and isinstance(r_dim, int) and l_dim != r_dim:
                raise ValueError(
                    f"Cannot stack shapes with different dimensions: {lhs} vs {rhs}"
                )
        out = [lhs[0], 2, *lhs[1:]]
        return out

    def _collect_plugin_value_info(
        self,
        graph: OnnxGraph,
        orig_node: NodeProto,
        plugin_op: NodeProto,
        q_shape: list[int | str],
        q_num_heads: int,
        head_size: int,
        q_input_name: str,
    ) -> None:
        """
        Collect value_info for AttentionPlugin outputs.

        Since AttentionPlugin is a custom TensorRT domain op, ONNX shape inferencer
        cannot recognize it. This method collects output value_info and defers
        update to a rewriter posthook.

        Args:
            graph: The ONNX computation graph
            orig_node: Original ONNX Attention node
            plugin_op: New AttentionPlugin node
        """
        # Output[0]: attn_output
        if len(plugin_op.output) > 0 and plugin_op.output[0]:
            attn_out_name = plugin_op.output[0]
            orig_attn_out_name = orig_node.output[0]
            attn_shape, attn_dtype = graph.tensor_info(orig_attn_out_name)
            if attn_shape is None:
                attn_shape = self._infer_attention_output_shape(
                    q_shape, q_num_heads, head_size
                )
            if attn_dtype == onnx.TensorProto.UNDEFINED:
                _, q_dtype = graph.tensor_info(q_input_name)
                attn_dtype = q_dtype
            if attn_dtype == onnx.TensorProto.UNDEFINED:
                return
            self._pending_value_infos[attn_out_name] = (attn_shape, attn_dtype)

        # Output[1]: present_key_value from merged KV cache output
        # Shape: [batch_size, 2, heads, seq_len, head_size]
        if len(plugin_op.output) > 1 and plugin_op.output[1]:
            kv_out_name = plugin_op.output[1]
            # Infer from original present_key if available
            if len(orig_node.output) > 2 and orig_node.output[1]:
                orig_key_name = orig_node.output[1]
                key_shape, key_dtype = graph.tensor_info(orig_key_name)
                # key_shape is [batch, seq, heads, head_dim]
                # merged shape is [batch, 2, heads, head_dim] (2 for key+value)
                merged_shape = self._stack_shape_axis1(key_shape, key_shape)
                if key_dtype != onnx.TensorProto.UNDEFINED:
                    self._pending_value_infos[kv_out_name] = (merged_shape, key_dtype)

    def _reset_pending_value_info(self, graph: OnnxGraph) -> OnnxGraph:
        self._pending_value_infos.clear()
        return graph

    def _flush_pending_value_info(self, graph: OnnxGraph) -> OnnxGraph:
        for name, (shape, dtype) in self._pending_value_infos.items():
            graph.set_value_info(name, shape, dtype)
        self._pending_value_infos.clear()
        return graph

    def _infer_attention_output_shape(
        self,
        q_shape: list[int | str],
        q_num_heads: int,
        head_size: int,
    ) -> list[int | str]:
        """Infer Attention output shape as [batch, seq, hidden]."""
        if len(q_shape) == 4:
            hidden: int | str = "hidden_size"
            heads = q_shape[1]
            dim = q_shape[3]
            if isinstance(heads, int) and isinstance(dim, int):
                hidden = heads * dim
            elif isinstance(q_num_heads, int) and q_num_heads > 0 and head_size > 0:
                hidden = q_num_heads * head_size
            return [q_shape[0], q_shape[2], hidden]
        if len(q_shape) == 3:
            return [q_shape[0], q_shape[1], q_shape[2]]
        return ["batch_size", "seq_len", "hidden_size"]
