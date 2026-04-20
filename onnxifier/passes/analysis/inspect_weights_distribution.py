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

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from onnx import NodeProto, TensorProto

from ... import OnnxGraph, logger
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter

# Bytes per element for each ONNX data type
_DTYPE_SIZE: dict[int, float] = {
    TensorProto.FLOAT: 4,
    TensorProto.UINT8: 1,
    TensorProto.INT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.BFLOAT16: 2,
    TensorProto.FLOAT8E4M3FN: 1,
    TensorProto.FLOAT8E4M3FNUZ: 1,
    TensorProto.FLOAT8E5M2: 1,
    TensorProto.FLOAT8E5M2FNUZ: 1,
    TensorProto.INT4: 0.5,
    TensorProto.UINT4: 0.5,
}


def _format_size(size_bytes: float) -> str:
    if size_bytes >= (1 << 30):
        return f"{size_bytes / (1 << 30):.2f} GiB"
    if size_bytes >= (1 << 20):
        return f"{size_bytes / (1 << 20):.2f} MiB"
    if size_bytes >= (1 << 10):
        return f"{size_bytes / (1 << 10):.2f} KiB"
    return f"{size_bytes:.0f} B"


def _weight_byte_size(tensor: TensorProto) -> float:
    num_elements = int(np.prod(tensor.dims)) if tensor.dims else 1
    elem_size = _DTYPE_SIZE.get(tensor.data_type, 1)
    return num_elements * elem_size


@PASSES.register(name="inspect_weights_distribution")
class InspectWeightsDistribution(Rewriter):
    """Inspect the weight size and occupancy percentage grouped by op_type."""

    def __init__(self, save_path: str | None = None):
        super().__init__(SingleNodePattern())
        self.save_path: str | None = save_path
        # graph id -> { op_type -> { weight_name: size_in_bytes } }
        self._stats: dict[int, dict[str, dict[str, float]]] = {}
        # graph id -> total initializer bytes (computed once)
        self._total_weight_size: dict[int, float] = {}
        self.register_post_hook(self._report)

    @staticmethod
    def _constant_node_byte_size(node: NodeProto) -> float:
        """Get byte size from a Constant node's tensor attribute."""
        for attr in node.attribute:
            if attr.name == "value" and attr.t.ByteSize() > 0:
                return _weight_byte_size(attr.t)
        return 0

    def _compute_total_weight_size(self, graph: OnnxGraph) -> float:
        gid = id(graph)
        if gid not in self._total_weight_size:
            total = sum(_weight_byte_size(init) for init in graph.initializer)
            for name in graph:
                pb: NodeProto = graph.nodes[name]["pb"]
                if pb.op_type == "Constant":
                    total += self._constant_node_byte_size(pb)
            self._total_weight_size[gid] = total
        return self._total_weight_size[gid]

    def _report(self, graph: OnnxGraph) -> OnnxGraph:
        gid = id(graph)
        stats = self._stats.get(gid)
        if not stats:
            return graph

        total_weight = self._compute_total_weight_size(graph)
        if total_weight == 0:
            logger.info("No weights found in graph.")
            return graph

        report: dict[str, dict[str, str | float]] = {}
        logger.info("=" * 60)
        logger.info("Weights Distribution Report")
        logger.info("=" * 60)
        logger.info(f"Total weight size: {_format_size(total_weight)}")
        logger.info("-" * 60)

        for op_type in sorted(
            stats, key=lambda t: sum(stats[t].values()), reverse=True
        ):
            weights = stats[op_type]
            group_size = sum(weights.values())
            pct = group_size / total_weight * 100 if total_weight else 0
            logger.info(
                f"  {op_type}: {_format_size(group_size)} "
                f"({pct:.2f}%, {len(weights)} weight(s))"
            )
            report[op_type] = {
                "size_bytes": group_size,
                "size_human": _format_size(group_size),
                "percentage": round(pct, 2),
                "num_weights": len(weights),
            }

        logger.info("=" * 60)

        if self.save_path:
            out = {
                "total_weight_bytes": total_weight,
                "total_weight_human": _format_size(total_weight),
                "by_op_type": report,
            }
            Path(self.save_path).write_text(
                json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(f"Report saved to {self.save_path}")

        # cleanup
        self._stats.pop(gid, None)
        self._total_weight_size.pop(gid, None)
        return graph

    def rewrite(
        self,
        graph: OnnxGraph,
        nodes: list[NodeProto],
        save_path: str | None = None,
    ):
        if save_path is not None:
            self.save_path = save_path
        gid = id(graph)
        if gid not in self._stats:
            self._stats[gid] = defaultdict(dict)
        stats = self._stats[gid]

        node = nodes[0]
        if node.op_type == "Constant":
            return

        initializers = graph.initializers
        for inp_name in node.input:
            size = 0.0
            if inp_name in initializers:
                size = _weight_byte_size(initializers[inp_name])
            else:
                inp_node = self.get_input_node(node, inp_name)
                if inp_node is not None and inp_node.op_type == "Constant":
                    size = self._constant_node_byte_size(inp_node)
            if size > 0:
                stats[node.op_type][inp_name] = size
