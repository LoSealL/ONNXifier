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

# pylint: disable=missing-function-docstring,redefined-outer-name

import numpy as np
import onnx
import pytest
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array

from onnxifier import ONNXIFIER_IR_VERSION, OnnxGraph, PassManager
from onnxifier.evaluator import Evaluator


def _make_attention_subgraph(
    batch_size: int = 1,
    seq_len: int = 10,
    num_heads: int = 4,
    head_size: int = 16,
    q_num_heads: int | None = None,  # pylint: disable=unused-argument
    kv_num_heads: int | None = None,  # pylint: disable=unused-argument
    with_bias: bool = False,  # pylint: disable=unused-argument
    with_mask: bool = False,
    opset: int = 24,
):
    """Create a minimal attention subgraph matching the FuseAttentionRewriter pattern.

    Pattern:
        input
         |---------> Q_proj (MatMul) --> Q
         |---------> K_proj (MatMul) --> K
         |---------> V_proj (MatMul) --> V
         |
         Q, K --> QK_matmul (MatMul) --> scale (Mul)
              --> Softmax --> SV_matmul (MatMul) --> output
    """
    # Inputs: [batch, num_heads, seq_len, head_size]
    hidden_size = head_size
    input_info = make_tensor_value_info(
        "input", TensorProto.FLOAT, [batch_size, num_heads, seq_len, hidden_size]
    )

    # Q, K, V projection weights [head_size, head_size]
    q_weight = from_array(
        np.random.randn(hidden_size, hidden_size).astype(np.float32), "q_weight"
    )
    k_weight = from_array(
        np.random.randn(hidden_size, hidden_size).astype(np.float32), "k_weight"
    )
    v_weight = from_array(
        np.random.randn(hidden_size, hidden_size).astype(np.float32), "v_weight"
    )

    # Q, K, V projection MatMuls
    q_matmul = make_node("MatMul", ["input", "q_weight"], ["q_out"], name="q_matmul")
    k_matmul = make_node("MatMul", ["input", "k_weight"], ["k_out"], name="k_matmul")
    v_matmul = make_node("MatMul", ["input", "v_weight"], ["v_out"], name="v_matmul")

    k_transpose = make_node(
        "Transpose", ["k_out"], ["k_t"], name="k_transpose", perm=[0, 1, 3, 2]
    )

    # Q @ K^T
    qk_matmul = make_node("MatMul", ["q_out", "k_t"], ["qk_scores"], name="qk_matmul")

    # Scale: QK * scale
    scale_val = 1.0 / np.sqrt(head_size)
    scale_const = from_array(np.array(scale_val, dtype=np.float32), "scale")
    scale_mul = make_node(
        "Mul", ["qk_scores", "scale"], ["scaled_qk"], name="scale_mul"
    )

    # Optional mask
    softmax_input = "scaled_qk"
    mask_add = None
    mask_init = None
    if with_mask:
        mask_init = from_array(
            np.ones((seq_len, seq_len), dtype=np.float32) * -1e9,
            "mask",
        )
        mask_add = make_node(
            "Add", ["scaled_qk", "mask"], ["masked_qk"], name="mask_add"
        )
        softmax_input = "masked_qk"

    # Softmax
    softmax = make_node("Softmax", [softmax_input], ["attn_weights"], name="softmax")

    # Softmax @ V
    sv_matmul = make_node(
        "MatMul", ["attn_weights", "v_out"], ["output"], name="sv_matmul"
    )

    outputs = [
        make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_size]
        )
    ]

    nodes = [q_matmul, k_matmul, v_matmul, k_transpose, qk_matmul, scale_mul]
    if mask_add is not None:
        nodes.extend([mask_add, softmax, sv_matmul])
    else:
        nodes.extend([softmax, sv_matmul])

    initializers = [q_weight, k_weight, v_weight, scale_const]

    if mask_init is not None:
        initializers.append(mask_init)

    graph = make_graph(
        nodes,
        "attention_graph",
        [input_info],
        outputs,
        initializer=initializers,
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[make_operatorsetid("", opset)],
    )
    onnx.checker.check_model(model)
    return OnnxGraph(model)


def test_fuse_attention_basic():
    """Test basic attention subgraph fusion."""
    graph = _make_attention_subgraph()
    runner1 = Evaluator(graph.model, "onnx")

    original_node_count = len(graph)

    graph = PassManager(["fuse_attention"]).optimize(graph, strict=True)

    # After fusion, we should have fewer nodes
    assert len(graph) < original_node_count

    # Check that an Attention node was created
    attention_found = False
    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type == "Attention":
            attention_found = True
            # Verify required attributes are set
            attr_names = [a.name for a in node.attribute]
            assert "q_num_heads" in attr_names
            assert "kv_num_heads" in attr_names
            assert "scale" in attr_names
            assert "head_size" not in attr_names
            break

    assert attention_found, "No Attention node found after fusion"

    runner2 = Evaluator(graph.model, "onnx")
    x = np.random.randn(1, 4, 10, 16).astype(np.float32)
    y1 = runner1(["output"], {"input": x})[0]
    y2 = runner2(["output"], {"input": x})[0]
    np.testing.assert_allclose(y1, y2, rtol=1e-5, atol=1e-5)


def test_skip_on_low_opset():
    """Test that fusion is skipped on opset < 23."""
    graph = _make_attention_subgraph(opset=22)

    graph = PassManager(["fuse_attention"]).optimize(graph, strict=True)

    # Should have original nodes since fusion was skipped
    for n in graph:
        node = graph.nodes[n]["pb"]
        assert node.op_type != "Attention"


def test_fuse_attention_with_mask():
    """Test attention fusion with attention mask."""
    graph = _make_attention_subgraph(with_mask=True)
    runner1 = Evaluator(graph.model, "onnx")

    graph = PassManager(["fuse_attention"]).optimize(graph, strict=True)

    # Should have fused successfully
    attention_found = False
    for n in graph:
        node = graph.nodes[n]["pb"]
        if node.op_type == "Attention":
            attention_found = True
            assert len(node.input) >= 4
            assert node.input[3] == "mask"
            break

    assert attention_found

    runner2 = Evaluator(graph.model, "onnx")
    x = np.random.randn(1, 4, 10, 16).astype(np.float32)
    y1 = runner1(["output"], {"input": x})[0]
    y2 = runner2(["output"], {"input": x})[0]
    np.testing.assert_allclose(y1, y2, rtol=1e-5, atol=1e-5)


@pytest.fixture(name="eager_attention_runners")
def _eager_attention_runners(tmp_path):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        pytest.skip("Requires PyTorch to run")

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def eager_attention_forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        num_key_value_groups: int = 1,
    ):
        key_states = repeat_kv(key, num_key_value_groups)
        value_states = repeat_kv(value, num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    class DummyAttn(torch.nn.Module):
        def __init__(self, q_head: int, kv_head: int, head_size: int):
            super().__init__()
            self.q_head = q_head
            self.kv_head = kv_head
            self.head_size = head_size

        def forward(self, q, k, v, attn_mask=None):
            return eager_attention_forward(
                q,
                k,
                v,
                attn_mask,
                scaling=1.0 / np.sqrt(self.head_size),
                num_key_value_groups=self.q_head // self.kv_head,
            )[0]

    def _build(with_mask: bool):
        model_path = (
            tmp_path / "eager_attn.onnx"
            if with_mask
            else tmp_path / "eager_attn_nomask.onnx"
        )

        args = [
            torch.randn(1, 4, 10, 16),
            torch.randn(1, 2, 10, 16),
            torch.randn(1, 2, 10, 16),
        ]
        input_names = ["q", "k", "v"]
        if with_mask:
            args.append(torch.randn(1, 1, 10, 10))
            input_names.append("mask")

        torch.onnx.export(
            DummyAttn(q_head=4, kv_head=2, head_size=16),
            tuple(args),
            model_path,
            input_names=input_names,
            output_names=["y"],
            opset_version=23,
            dynamo=True,
        )

        graph = OnnxGraph(onnx.load(model_path))
        runner1 = Evaluator(graph.model, "onnx")

        graph = PassManager(["fuse_attention"]).optimize(graph, strict=True)
        runner2 = Evaluator(graph.model, "onnx")

        found_attention = sum(
            1 for n in graph if graph.nodes[n]["pb"].op_type == "Attention"
        )
        assert found_attention == 1
        return runner1, runner2

    return _build


@pytest.mark.parametrize("with_mask", [True, False], ids=["mask", "no-mask"])
def test_eager_attention_forward(eager_attention_runners, with_mask):
    runner1, runner2 = eager_attention_runners(with_mask=with_mask)

    q = np.random.randn(1, 4, 10, 16).astype(np.float32)
    k = np.random.randn(1, 2, 10, 16).astype(np.float32)
    v = np.random.randn(1, 2, 10, 16).astype(np.float32)
    feeds = {"q": q, "k": k, "v": v}
    if with_mask:
        # ReferenceEvaluator's Attention implementation cannot consume 4D float
        # mask in this environment, while unfused Add supports 2D broadcast.
        feeds["mask"] = np.random.randn(10, 10).astype(np.float32)

    y1 = runner1(["y"], feeds)[0]
    y2 = runner2(["y"], feeds)[0]
    np.testing.assert_allclose(y1, y2, rtol=1e-5, atol=1e-5)
