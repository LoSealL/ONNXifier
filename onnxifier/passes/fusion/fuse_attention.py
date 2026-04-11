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
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("fuse_attention", deps=["infer_shape"])
class FuseAttentionRewriter(Rewriter):
    """Fuse scaled-dot-product attention core into ONNX Attention op.

    Pattern (mask optional):
      Q,K -> MatMul -> Mul(scale) -> [Add(mask)] -> Softmax
      Softmax,V -> MatMul -> Y

    Q/K/V are expected to be outputs of projection branches and are passed
    directly to Attention inputs.
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Softmax"))

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto]):
        # Attention op is available since opset 23.
        # Restrict to 23+ for current pass behavior and tests.
        if graph.opset_version < 23:
            return

        softmax = nodes[0]
        if len(softmax.input) != 1 or len(softmax.output) != 1:
            return

        pre_node = self.get_input_node(softmax, 0)
        if pre_node is None:
            return

        mask_input = ""
        scale_mul = None

        if pre_node.op_type == "Add":
            scale_mul, mask_input = self._parse_mask_add(pre_node)
            if scale_mul is None:
                return
        elif pre_node.op_type == "Mul":
            scale_mul = pre_node
        else:
            return

        qk_matmul, scale = self._parse_scale_mul(scale_mul)
        if qk_matmul is None or scale is None:
            return

        q_tensor, k_tensor = self._parse_qk_inputs(qk_matmul)
        if not q_tensor or not k_tensor:
            return

        sv_matmul, v_tensor = self._find_sv_matmul(softmax)
        if sv_matmul is None or not v_tensor:
            return

        q_heads, kv_heads = self._infer_heads(q_tensor, k_tensor, scale)
        if q_heads is None or kv_heads is None:
            return

        if not self._is_single_consumer(softmax, softmax.output[0], sv_matmul.name):
            return
        if not self._is_single_consumer(qk_matmul, qk_matmul.output[0], scale_mul.name):
            return
        if pre_node.op_type == "Add":
            if not self._is_single_consumer(
                scale_mul, scale_mul.output[0], pre_node.name
            ):
                return
            if not self._is_single_consumer(pre_node, pre_node.output[0], softmax.name):
                return
        else:
            if not self._is_single_consumer(
                scale_mul, scale_mul.output[0], softmax.name
            ):
                return

        attn_inputs = [q_tensor, k_tensor, v_tensor]
        if mask_input:
            attn_inputs.append(mask_input)

        attention_node = make_node(
            "Attention",
            inputs=attn_inputs,
            outputs=[sv_matmul.output[0]],
            name=f"{sv_matmul.name}/Attention",
            q_num_heads=q_heads,
            kv_num_heads=kv_heads,
            scale=float(scale),
            is_causal=0,
        )

        self += attention_node
        self -= [qk_matmul, scale_mul, softmax, sv_matmul]
        if pre_node.op_type == "Add":
            self -= pre_node

    def _parse_mask_add(self, add_node: NodeProto) -> tuple[NodeProto | None, str]:
        if len(add_node.input) != 2:
            return None, ""
        left = self.get_input_node(add_node, add_node.input[0])
        right = self.get_input_node(add_node, add_node.input[1])
        if left is not None and left.op_type == "Mul":
            return left, add_node.input[1]
        if right is not None and right.op_type == "Mul":
            return right, add_node.input[0]
        return None, ""

    def _parse_scale_mul(
        self, mul_node: NodeProto
    ) -> tuple[NodeProto | None, float | None]:
        if len(mul_node.input) != 2:
            return None, None
        left = self.get_input_node(mul_node, mul_node.input[0])
        right = self.get_input_node(mul_node, mul_node.input[1])

        if left is not None and left.op_type == "MatMul":
            scale = self._const_scalar(mul_node.input[1])
            return left, scale
        if right is not None and right.op_type == "MatMul":
            scale = self._const_scalar(mul_node.input[0])
            return right, scale
        return None, None

    def _parse_qk_inputs(self, qk_matmul: NodeProto) -> tuple[str, str]:
        if len(qk_matmul.input) != 2:
            return "", ""

        a, b = qk_matmul.input
        a_node = self.get_input_node(qk_matmul, a)
        b_node = self.get_input_node(qk_matmul, b)

        # Typical form is MatMul(Q, Transpose(K)); Q may come from graph input
        # or any projection branch, so only require one side to be Transpose.
        if b_node is not None and b_node.op_type == "Transpose":
            k = self._parse_k_from_transpose(b_node)
            return a, k
        if a_node is not None and a_node.op_type == "Transpose":
            k = self._parse_k_from_transpose(a_node)
            return b, k
        return "", ""

    @staticmethod
    def _parse_k_from_transpose(transpose: NodeProto) -> str:
        if len(transpose.input) != 1:
            return ""
        return transpose.input[0]

    def _find_sv_matmul(self, softmax: NodeProto) -> tuple[NodeProto | None, str]:
        consumers = self.get_output_node(softmax, softmax.output[0])
        for node in consumers:
            if node.op_type != "MatMul" or len(node.input) != 2:
                continue
            if node.input[0] == softmax.output[0]:
                return node, node.input[1]
            if node.input[1] == softmax.output[0]:
                return node, node.input[0]
        return None, ""

    def _infer_heads(
        self, q_tensor: str, k_tensor: str, scale: float
    ) -> tuple[int | None, int | None]:
        q_shape = self.graph.tensor_shape(q_tensor)
        k_shape = self.graph.tensor_shape(k_tensor)
        if len(q_shape) == 4 and len(k_shape) == 4:
            if isinstance(q_shape[1], int) and isinstance(k_shape[1], int):
                return q_shape[1], k_shape[1]
            return None, None

        # For 3D tensors, this subgraph has no explicit head-split operators.
        # It is semantically equivalent to Attention only when heads == 1.
        if len(q_shape) != 3 or len(k_shape) != 3:
            return None, None
        q_hidden = q_shape[2]
        k_hidden = k_shape[2]
        if not isinstance(q_hidden, int) or not isinstance(k_hidden, int):
            return None, None
        if scale <= 0:
            return None, None

        head_size_f = (1.0 / scale) ** 2
        head_size = int(round(head_size_f))
        if (
            head_size <= 0
            or not np.isclose(head_size_f, head_size, rtol=1e-4, atol=1e-4)
            or q_hidden != head_size
            or k_hidden != head_size
        ):
            return None, None
        return 1, 1

    def _const_scalar(self, name: str) -> float | None:
        value = self.get_value(name)
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.size != 1:
            return None
        return float(arr.reshape(-1)[0])

    def _is_single_consumer(
        self, node: NodeProto, output_name: str, consumer_name: str
    ) -> bool:
        consumers = self.get_output_node(node, output_name)
        return len(consumers) == 1 and consumers[0].name == consumer_name
