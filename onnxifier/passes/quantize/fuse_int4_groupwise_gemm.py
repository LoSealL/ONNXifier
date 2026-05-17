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
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ...domain.trt.ops.int4_gemm_plugin import int4_gemm_plugin_schema
from ...graph import OnnxGraph
from ...logger import debug
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(
    "fuse_int4_groupwise_gemm",
    deps=["infer_shape", "trt_dequantize_linear_to_onnx"],
)
class FuseInt4GroupwiseGemmRewriter(Rewriter):
    """Fuse DequantizeLinear -> (** ->) MatMul into Int4GroupwiseGemmPlugin.

    The pass runs after ``trt_dequantize_linear_to_onnx``, which converts
    ``trt::DequantizeLinear`` back to ONNX ``DequantizeLinear`` (domain ``""``).
    Only two layout patterns are supported:

    - **axis=0, non-transposed**: ``DequantizeLinear[axis=0] → MatMul``
      Weight is ``[K, N]`` quantized along axis 0 (K dimension).
    - **axis=1, transposed**: ``DequantizeLinear[axis=1] → Transpose → MatMul``
      Weight is ``[K, N]`` quantized along axis 1 (N dimension), then
      transposed to ``[N, K]`` for MatMul consumption.

    Before (axis=0):

        DequantizeLinear[axis=0] → MatMul ← activation

    Before (axis=1):

        DequantizeLinear[axis=1] → Transpose → MatMul ← activation

    After:

        Int4GroupwiseGemmPlugin(activation, qweight_packed, scales)
    """

    def __init__(self):
        super().__init__(SingleNodePattern("MatMul"))

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto], *_a, **_kw):
        node = nodes[0]

        for weight_idx, inp_name in enumerate(node.input):
            dq_node, path_nodes = self._walk_to_dq(inp_name)
            if dq_node is None:
                continue

            act_idx = 1 - weight_idx
            try:
                weight_shape = graph.static_tensor_shape(node.input[weight_idx])
            except ValueError:
                debug("can't get weight shape of %s", node.name)
                return
            # Determine whether the weight path includes a 2D transpose.
            # Transposed path: DQ[axis=1] → (Reshape →) Transpose[1,0] → MatMul
            # Non-transposed path: DQ[axis=0] → MatMul
            transposed = self._has_transpose(path_nodes)

            gemm_k = weight_shape[-2]
            gemm_n = weight_shape[-1]
            block_size = self.get_attribute(dq_node, "block_size", 0)
            axis = self.get_attribute(dq_node, "axis", 1)
            qweight = self.get_value_or_die(dq_node.input[0])
            scales = self.get_value_or_die(dq_node.input[1])

            # Prepare qweight for _pack_int4 which expects (N, K) layout.
            # _pack_int4 packs along N (output features) with interleave/stride.
            if transposed:
                # axis=1 required: DQ scales along N dim of [K, N] weight.
                # qweight arrives as [K, N], reshape to [N, K] for packing.
                # scales: [K, N//block_size] → reshape to [K//block_size, N]
                assert axis == 1
                qweight = qweight.reshape([gemm_n, gemm_k])
                scales = scales.reshape([-1, gemm_k // block_size]).T
            else:
                # axis=0 required: DQ scales along K dim of [K, N] weight.
                # qweight arrives as [K, N], transpose to [N, K] for packing.
                # scales: [K//block_size, N] — already in final layout.
                assert axis == 0
                qweight = qweight.reshape([gemm_k, gemm_n]).T
                scales = scales.reshape([gemm_k // block_size, -1])
            packed = _pack_int4(qweight)
            # Final scales shape: [gemm_k // block_size, gemm_n]
            scales = scales.reshape([gemm_k // block_size, gemm_n])

            packed_const = make_constant(f"{node.name}/qweight_packed", packed)
            scales_const = make_constant(f"{node.name}/scales", scales)

            plugin = make_node(
                int4_gemm_plugin_schema.name,
                inputs=[
                    node.input[act_idx],
                    packed_const.output[0],
                    scales_const.output[0],
                ],
                outputs=[node.output[0]],
                domain=int4_gemm_plugin_schema.domain,
                name=f"{node.name}_int4",
                gemm_n=gemm_n,
                gemm_k=gemm_k,
                group_size=block_size,
            )
            self += [plugin, packed_const, scales_const]
            self -= [node, dq_node, *path_nodes]
            return

    def _walk_to_dq(self, tensor_name: str) -> tuple[NodeProto | None, list[NodeProto]]:
        """Walk upstream from tensor_name to find a DequantizeLinear producer.

        After ``trt_dequantize_linear_to_onnx`` runs, the DQ node has
        domain ``""`` (standard ONNX), not ``"trt"``.  Intermediate
        Transpose or Reshape nodes on the path are collected as
        ``path_nodes``.

        Returns (dq_node, intermediate_nodes_on_path) or (None, []).
        """
        graph = self.graph
        # pylint: disable=protected-access
        producer_name = graph._out_to_node.get(tensor_name)
        if producer_name is None:
            return None, []

        producer: NodeProto = graph.nodes[producer_name]["pb"]

        if producer.op_type == "DequantizeLinear":
            return producer, []

        if producer.op_type in ("Transpose", "Reshape"):
            for inp in producer.input[:1]:
                dq, path = self._walk_to_dq(inp)
                if dq is not None:
                    return dq, [producer, *path]

        return None, []

    @staticmethod
    def _has_transpose(path_nodes: list[NodeProto]) -> bool:
        for n in path_nodes:
            if n.op_type == "Transpose":
                for a in n.attribute:
                    if a.name == "perm" and list(a.ints) == [1, 0]:
                        return True
        return False


def _pack_int4(qweight: np.ndarray) -> np.ndarray:
    """Pack int4 weight values into interleaved int8 storage for GPU kernels.

    The packing follows the GPTQ kernel layout used by TensorRT:

    1. Shift int4 values from [-8,7] to unsigned [0,15] (``+8``).
    2. Reorder within each 32-element K-block: swap inner (4×4×2) groups
       so that consecutive weight pairs align for 4-bit nibble packing.
    3. Interleave every 4 rows (``interleave=4``): rows 0-3 are grouped
       into a single super-row so the GPU can dequantize 4 rows at once.
    4. Pack 4 unsigned 4-bit values into one int16:  ``v0 | v1<<4 | v2<<8 | v3<<12``.
    5. View the int16 result as int8, halving the row dimension.

    Parameters
    ----------
    qweight : np.ndarray
        Signed int4 weight values in range ``[-8, 7]`` with shape ``(N, K)``.
        ``N`` is the output-feature dimension (rows) and ``K`` is the
        input-feature dimension (columns).  Both must be divisible by
        ``interleave=4`` and ``kstride=64`` (so ``K`` must be a multiple of 64).

    Returns
    -------
    np.ndarray
        Packed weight with dtype ``int8`` and shape ``(N//2, K)``.
        Each int8 byte stores one nibble pair; the logical layout is
        ``(N//4, K)`` in int16 where each int16 holds 4 nibbles.
    """
    interleave = 4
    kstride = 64
    n = qweight.shape[0]
    k = qweight.shape[1]

    # Shift signed int4 [-8,7] to unsigned [0,15] for nibble packing
    qweight = qweight.astype(np.int16) + 8

    # Split K into blocks of 32 and reorder within each block
    # Original layout: (N, K//32, 4, 4, 2) — 4 groups of 4×2 elements
    # Swap inner groups: transpose dims 2↔3 so pairs align for nibbles
    packed = qweight.reshape(n, k // 32, 32)
    packed = packed.reshape(n, k // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    # Reorder 8-weight micro-blocks for fast GPU dequantization stride
    packed = packed.reshape(n, k // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)

    # Interleave every 4 rows: group (N//4, 4) then reorder to
    # (N//4, K//64, 4, 64) so rows 0-3 share a super-row
    packed = packed.reshape(n // interleave, interleave, k // kstride, kstride)
    packed = packed.transpose(0, 2, 1, 3)
    packed = packed.reshape(n // interleave, k // kstride, kstride, interleave)

    # Pack 4 unsigned 4-bit values into one int16:
    #   nibble0 (row0) | nibble1 (row1)<<4 | nibble2 (row2)<<8 | nibble3 (row3)<<12
    packed = (
        packed[..., 0]
        | (packed[..., 1] << 4)
        | (packed[..., 2] << 8)
        | (packed[..., 3] << 12)
    )

    # Final layout: (N//4, K) as int16, viewed as (N//2, K) int8
    return packed.reshape(n // interleave, k).view(np.int8).reshape(n // 2, k)
