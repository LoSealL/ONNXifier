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
from ...logger import debug, warning
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register("fuse_int4_groupwise_gemm", deps=["infer_shape"])
class FuseInt4GroupwiseGemmRewriter(Rewriter):
    """Fuse DequantizeLinear -> ** -> MatMul into Int4GroupwiseGemmPlugin.

    Before:

        DequantizeLinear
              |  (weight path, possibly through Transpose/Reshape)
              v
           MatMul  <-- activation

    After:

        Int4GroupwiseGemmPlugin(input, qweight_packed, scales)
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

            gemm_k = weight_shape[-2]
            gemm_n = weight_shape[-1]
            block_size = self.get_attribute(dq_node, "block_size", 0)
            axis = self.get_attribute(dq_node, "axis", 1)

            qweight_float = self.get_value_or_die(dq_node.input[0])
            scales_float = self.get_value_or_die(dq_node.input[1])
            packed = _pack_int4(dq_node, qweight_float, scales_float, block_size, axis)
            packed_const = make_constant(f"{node.name}/qweight_packed", packed)

            plugin = make_node(
                int4_gemm_plugin_schema.name,
                inputs=[
                    node.input[act_idx],
                    packed_const.output[0],
                    dq_node.input[1],
                ],
                outputs=[node.output[0]],
                domain=int4_gemm_plugin_schema.domain,
                name=f"{node.name}_int4",
                gemm_n=gemm_n,
                gemm_k=gemm_k,
                group_size=block_size,
            )
            self += [plugin, packed_const]
            self -= [node, dq_node, *path_nodes]
            return

    def _walk_to_dq(self, tensor_name: str) -> tuple[NodeProto | None, list[NodeProto]]:
        """Walk upstream from tensor_name to find a trt::DequantizeLinear.

        Returns (dq_node, intermediate_nodes_on_path) or (None, []).
        """
        graph = self.graph
        # pylint: disable=protected-access
        producer_name = graph._out_to_node.get(tensor_name)
        if producer_name is None:
            return None, []

        producer: NodeProto = graph.nodes[producer_name]["pb"]

        if producer.op_type == "DequantizeLinear" and producer.domain == "trt":
            return producer, []

        if producer.op_type in ("Transpose", "Reshape"):
            for inp in producer.input[:1]:
                dq, path = self._walk_to_dq(inp)
                if dq is not None:
                    return dq, [producer, *path]

        return None, []


def _pack_int4(
    node: NodeProto,
    weight: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    axis: int = 0,
) -> np.ndarray:
    """Quantize pseudo-quantized float weight to int4 and pack to int8.

    Packing is done along the quantization axis: two consecutive int4 values
    along axis are stored in one int8 byte (low nibble = first element,
    high nibble = second element).
    """
    if group_size > 0 and scales.shape[axis] * group_size == weight.shape[axis]:
        scales = np.repeat(scales, group_size, axis=axis)

    zero_mask = scales == 0
    near_zero_mask = np.abs(scales) < np.finfo(scales.dtype).tiny
    if zero_mask.any() or near_zero_mask.any():
        count = int(zero_mask.sum() + near_zero_mask.sum())
        warning(
            "node %s: %d scale values are zero or near-zero; "
            "int4 quantization may produce Inf/NaN values",
            node.name,
            count,
        )

    int_vals = np.rint(weight / scales).astype(np.int32)
    int_vals = np.clip(int_vals, -8, 7).astype(np.int8)

    uint4 = (int_vals + 8).astype(np.uint8)

    dim_size = uint4.shape[axis]
    if dim_size % 2 != 0:
        raise ValueError(
            f"Dimension {axis} must be even for int4 packing, got {dim_size}"
        )

    even = np.take(uint4, np.arange(0, dim_size, 2), axis=axis)
    odd = np.take(uint4, np.arange(1, dim_size, 2), axis=axis)
    packed = ((odd << 4) | even).view(np.int8)

    return packed
