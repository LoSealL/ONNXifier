"""
Copyright 2026 The ONNXIFIER Authors

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
from onnx import NodeProto, helper, numpy_helper

from ...graph import OnnxGraph
from ...logger import debug
from .. import PASSES

# fp16 min normal (2^-14); reciprocals of smaller magnitudes overflow fp16.
FP16_MIN_NORMAL = np.float16(6.103515625e-05)
FP16_MAX = np.float16(65504.0)


def _constant_values(graph: OnnxGraph) -> dict[str, np.ndarray]:
    """Map tensor names to their fp16 constant values (nodes + initializers)."""
    consts: dict[str, np.ndarray] = {}
    for pb in graph.model.graph.node:
        if pb.op_type == "Constant":
            for attr in pb.attribute:
                if attr.name == "value":
                    consts[pb.output[0]] = numpy_helper.to_array(attr.t)
    for name, tensor in graph.initializers.items():
        consts.setdefault(name, numpy_helper.to_array(tensor))
    return consts


def _constant_values_of_nodes(nodes) -> dict[str, np.ndarray]:
    """Map tensor names to their constant values from ``Constant`` nodes."""
    consts: dict[str, np.ndarray] = {}
    for pb in nodes:
        if pb.op_type == "Constant":
            for attr in pb.attribute:
                if attr.name == "value":
                    consts[pb.output[0]] = numpy_helper.to_array(attr.t)
    return consts


def _normalizing_scale(arr: np.ndarray) -> np.float16 | None:
    """Smallest power-of-two that brings every nonzero |c| into fp16 normals.

    Returns None when no single scale can normalize the array without
    overflowing its largest magnitude (dynamic range beyond fp16 normals).
    """
    mag = np.abs(arr.astype(np.float32))
    nz = mag[(arr != 0) & (np.abs(arr) < float(FP16_MIN_NORMAL))]
    if nz.size == 0:
        return None
    # scale >= FP16_MIN_NORMAL / min|c|, rounded up to a power of two
    ratio = float(FP16_MIN_NORMAL) / float(nz.min())
    k = int(np.ceil(np.log2(ratio)))
    m = float(np.float32(2.0) ** k)
    if float(mag.max()) * m > float(FP16_MAX) / 2:
        return None
    return np.float16(m)


def _widen_div_nodes(
    nodes, consts: dict[str, np.ndarray]
) -> tuple[list[NodeProto], int]:
    """Rewrite fp16 ``Div`` by subnormal constants into normal-range fp16 ops.

    ``x / c`` becomes ``(x * m) / (c * m)`` with a power-of-two ``m`` that
    lifts every subnormal element of ``c`` into the fp16 normal range. The
    multiply by ``m`` is exact in fp16, and the widened divisor keeps
    ``1 / (c * m)`` representable, so the rewrite survives being folded into
    surrounding fp16 fusions (TensorRT non-strongly-typed FP16 mode ignores
    Cast-based fp32 islands on 10.16).
    """
    new_nodes: list[NodeProto] = []
    patched = 0
    for pb in nodes:
        if pb.op_type != "Div" or len(pb.input) != 2:
            new_nodes.append(pb)
            continue
        divisor = consts.get(pb.input[1])
        if divisor is None or divisor.dtype != np.float16:
            new_nodes.append(pb)
            continue
        arr = np.asarray(divisor)
        if not np.any((arr != 0) & (np.abs(arr) < float(FP16_MIN_NORMAL))):
            new_nodes.append(pb)
            continue
        m = _normalizing_scale(arr)
        if m is None or not np.isfinite(np.float32(m)):
            new_nodes.append(pb)
            continue
        output = pb.output[0]
        m_name = output + "_widen_m"
        cw_name = output + "_widen_c"
        new_nodes.append(
            helper.make_node(
                "Constant",
                [],
                [m_name],
                value=numpy_helper.from_array(np.array(m), m_name),
            )
        )
        new_nodes.append(
            helper.make_node(
                "Constant",
                [],
                [cw_name],
                value=numpy_helper.from_array(
                    (arr.astype(np.float32) * np.float32(m)).astype(np.float16),
                    cw_name,
                ),
            )
        )
        new_nodes.append(
            helper.make_node("Mul", [pb.input[0], m_name], [output + "_widen_mul"])
        )
        new_nodes.append(
            helper.make_node("Div", [output + "_widen_mul", cw_name], [output])
        )
        patched += 1
    return new_nodes, patched


@PASSES.register("widen_subnormal_scaling")
def widen_subnormal_scaling(graph: OnnxGraph) -> OnnxGraph:
    """Rewrite fp16 ``Div`` by subnormal constants into an fp32 division.

    TensorRT (at least 10.14-11.1) folds a ``Div`` by a constant into the
    surrounding Myelin fusion as ``Mul(1/c)`` evaluated in fp16. When ``c``
    is subnormal enough that ``1/c`` overflows fp16 (|c| < ~6.1e-5), the
    folded reciprocal becomes inf and the fused kernel emits NaN (e.g. 0*inf
    where the true quotient is 0). The division is rescaled to
    ``(x * m) / (c * m)`` with a power-of-two ``m`` that normalizes ``c``,
    which is exact in fp16 and survives fusion folding.

    Both the main graph and every local function body are rewritten; shared
    function bodies (called from multiple sites) are invisible to
    :class:`PassManager` recursion, so the pass handles them itself.

    Example:

        onnxifier model.onnx -a widen_subnormal_scaling
    """
    model = graph.model
    patched = 0
    new_nodes, n = _widen_div_nodes(model.graph.node, _constant_values(graph))
    if n:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        patched += n
    for function in model.functions:
        new_nodes, n = _widen_div_nodes(
            function.node, _constant_values_of_nodes(function.node)
        )
        if n:
            del function.node[:]
            function.node.extend(new_nodes)
            patched += n
    if patched:
        debug("widen_subnormal_scaling: patched %d Div nodes", patched)
        return OnnxGraph(model, base_dir=graph.external_base)
    return graph
