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

# pylint: disable=line-too-long
import numpy as np
from onnx import NodeProto, TensorProto
from onnx.helper import make_node, tensor_dtype_to_np_dtype

from ...utils import text_to_boolean
from .. import OnnxGraph, make_constant
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register("scaled_dot_product_attention")
class ScaledDotProductAttention(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/scaled-dot-product-attention.html

    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    Y = SDPA(X) = softmax(XW^T/sqrt(d_k))XW

    Ref:
        https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto):
        inputs = self.get_input_nodes(ori_node)
        attention_mask, scale = None, None
        if len(inputs) == 4:
            attention_mask = inputs[3]
        elif len(inputs) == 5:
            attention_mask, scale = inputs[3:]
        causal = self.get_attribute(ori_node, "causal")
        if not isinstance(causal, str):
            raise ValueError(
                f"{ori_node.name} causal should be a string, but got {causal}"
            )
        is_causal = text_to_boolean(causal)

        dtype = tensor_dtype_to_np_dtype(graph.tensor_type(ori_node.input[0]))

        if scale is None:
            query_dim = graph.tensor_shape(ori_node.input[0])[-1]
            assert isinstance(query_dim, int) and query_dim > 0
            scale_value = (1 / np.sqrt(query_dim)).astype(dtype)
            scale = make_constant(f"{ori_node.name}/scale", scale_value)
            self += scale
        attn_bias = np.zeros((1, 1), dtype=dtype)
        if is_causal:
            L = graph.static_tensor_shape(ori_node.input[0])[-2]
            S = graph.static_tensor_shape(ori_node.input[1])[-2]
            attn_bias = np.triu(np.broadcast_to(-np.inf, (L, S)), k=1).astype(dtype)
        bias = make_constant(f"{ori_node.name}/bias", attn_bias)
        bias_result = bias.output[0]
        if not is_causal and attention_mask is not None:
            mask_dtype = graph.tensor_type(ori_node.input[3])
            if mask_dtype == TensorProto.BOOL:
                raise NotImplementedError("bool mask is not supported yet")
            else:
                # bias += attention_mask
                add_mask = make_node(
                    "Add",
                    inputs=[bias.output[0], attention_mask.output[0]],
                    outputs=[f"{ori_node.name}/add_attn_mask_output0"],
                    name=f"{ori_node.name}/add_attn_mask",
                )
                self += add_mask
                bias_result = add_mask.output[0]

        key_rank = len(graph.tensor_shape(ori_node.input[1]))
        key_perm = list(range(key_rank))
        key_perm[-2], key_perm[-1] = key_perm[-1], key_perm[-2]
        trans_node = make_node(
            "Transpose",
            inputs=[ori_node.input[1]],
            outputs=[f"{ori_node.name}/key/transpose_output0"],
            perm=key_perm,
            name=f"{ori_node.name}/key/transpose",
        )
        gemm0 = make_node(
            "MatMul",
            inputs=[ori_node.input[0], trans_node.output[0]],
            outputs=[f"{ori_node.name}/gemm0_output0"],
            name=f"{ori_node.name}/gemm0",
        )
        dot_scale = make_node(
            "Mul",
            inputs=[gemm0.output[0], scale.output[0]],
            outputs=[f"{ori_node.name}/qk_scale_output0"],
            name=f"{ori_node.name}/qk_scale",
        )
        add_bias = make_node(
            "Add",
            inputs=[dot_scale.output[0], bias_result],
            outputs=[f"{ori_node.name}/attn_bias_output0"],
            name=f"{ori_node.name}/attn_bias",
        )
        act_node = make_node(
            "Softmax",
            inputs=[add_bias.output[0]],
            outputs=[f"{ori_node.name}/softmax_output0"],
            axis=-1,
            name=f"{ori_node.name}/softmax",
        )
        gemm1 = make_node(
            "MatMul",
            inputs=[act_node.output[0], ori_node.input[2]],
            outputs=[ori_node.output[0]],
            name=f"{ori_node.name}",
        )

        self += [bias, trans_node, gemm0, dot_scale, add_bias, act_node]
        return gemm1
