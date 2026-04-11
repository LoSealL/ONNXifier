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

import math

from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register(name="attention_fill_heads_and_dim", deps=["infer_shape"])
class AttentionFillHeadsAndDimRewriter(Rewriter):
    """Fill inferable optional attributes of ONNX Attention op.

    This pass focuses on attributes that can be reliably inferred from shapes
    and will overwrite existing values if they disagree with inference:
    - q_num_heads
    - kv_num_heads
    - scale (1 / sqrt(head_size))

    It also materializes default-valued attributes defined by ONNX spec:
    - is_causal = 0
    - qk_matmul_output_mode = 0
    - softcap = 0.0
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Attention"))

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto]):
        node = nodes[0]

        # Fill defaults from ONNX operator schema when missing.
        if self.get_attribute(node, "is_causal") is None:
            self.set_attribute(node, "is_causal", 0)
        if self.get_attribute(node, "qk_matmul_output_mode") is None:
            self.set_attribute(node, "qk_matmul_output_mode", 0)
        if self.get_attribute(node, "softcap") is None:
            self.set_attribute(node, "softcap", 0.0)

        q_num_heads_attr = self._as_positive_int(
            self.get_attribute(node, "q_num_heads")
        )
        kv_num_heads_attr = self._as_positive_int(
            self.get_attribute(node, "kv_num_heads")
        )

        q_shape = graph.tensor_shape(node.input[0])
        k_shape = graph.tensor_shape(node.input[1])
        past_k_shape = graph.tensor_shape(node.input[4]) if len(node.input) > 4 else []

        q_heads = None
        kv_heads = None
        head_size = None

        if len(q_shape) == 4:
            q_heads = self._as_positive_int(q_shape[1])
            head_size = self._pick_same(head_size, self._as_positive_int(q_shape[3]))
        q_hidden = self._as_positive_int(q_shape[2]) if len(q_shape) == 3 else None

        if len(k_shape) == 4:
            kv_heads = self._as_positive_int(k_shape[1])
            head_size = self._pick_same(head_size, self._as_positive_int(k_shape[3]))
        k_hidden = self._as_positive_int(k_shape[2]) if len(k_shape) == 3 else None

        if len(past_k_shape) == 4 and kv_heads is None:
            kv_heads = self._as_positive_int(past_k_shape[1])
            head_size = self._pick_same(
                head_size, self._as_positive_int(past_k_shape[3])
            )

        # Infer head size from known hidden-size / num-heads relation.
        q_heads_hint = q_heads or q_num_heads_attr
        kv_heads_hint = kv_heads or kv_num_heads_attr

        if (
            head_size is None
            and q_hidden
            and q_heads_hint
            and q_hidden % q_heads_hint == 0
        ):
            head_size = q_hidden // q_heads_hint
        if (
            head_size is None
            and k_hidden
            and kv_heads_hint
            and k_hidden % kv_heads_hint == 0
        ):
            head_size = k_hidden // kv_heads_hint

        # Infer head numbers from known head size.
        if q_heads is None and q_hidden and head_size and q_hidden % head_size == 0:
            q_heads = q_hidden // head_size
        if kv_heads is None and k_hidden and head_size and k_hidden % head_size == 0:
            kv_heads = k_hidden // head_size

        if q_heads is not None:
            self.set_attribute(node, "q_num_heads", q_heads)
        if kv_heads is not None:
            self.set_attribute(node, "kv_num_heads", kv_heads)

        # ONNX default: scale = 1 / sqrt(head_size). If head_size is known,
        # force-write to avoid keeping stale or incorrect existing values.
        if head_size and self.get_attribute(node, "scale") is None:
            self.set_attribute(node, "scale", 1.0 / math.sqrt(float(head_size)))

    @staticmethod
    def _as_positive_int(value) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value > 0:
            return value
        return None

    @staticmethod
    def _pick_same(current: int | None, new_value: int | None) -> int | None:
        if new_value is None:
            return current
        if current is None:
            return new_value
        return current if current == new_value else None
