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

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("fuse_gelu")
class FuseGeluRewriter(Rewriter):
    r"""Fuse erf-based GELU decomposition to Gelu op.

    Match this pattern:

        y = Mul(Add(Erf(Div(x, sqrt(2))), 1), Mul(x, 0.5))
    """

    def __init__(self):
        pattern = GraphPattern()
        node_x = SingleNodePattern()
        div = SingleNodePattern("Div")
        erf = SingleNodePattern("Erf")
        add = SingleNodePattern("Add")
        mul_half = SingleNodePattern("Mul")
        mul = SingleNodePattern("Mul")
        pattern.add_edge(node_x, div)
        pattern.add_edge(div, erf)
        pattern.add_edge(erf, add)
        pattern.add_edge(node_x, mul_half)
        pattern.add_edge(add, mul)
        pattern.add_edge(mul_half, mul)
        super().__init__(pattern)

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto]):
        # Gelu is introduced in ONNX opset 20.
        if graph.opset_version < 20:
            return

        node_x, div, erf, add, mul_half, mul = nodes
        x = node_x.output[0]

        div_const = self._other_input(div, x)
        if div_const is None or not self._is_close_scalar(
            self.get_value(div_const), math.sqrt(2.0)
        ):
            return

        erf_out = erf.output[0]
        add_const = self._other_input(add, erf_out)
        if add_const is None or not self._is_close_scalar(
            self.get_value(add_const), 1.0
        ):
            return

        mul_half_const = self._other_input(mul_half, x)
        if mul_half_const is None or not self._is_close_scalar(
            self.get_value(mul_half_const), 0.5
        ):
            return

        if set(mul.input) != {add.output[0], mul_half.output[0]}:
            return

        # Only fuse when intermediate nodes are dedicated to this GELU path.
        if not self._single_consumer(div, div.output[0], erf.name):
            return
        if not self._single_consumer(erf, erf.output[0], add.name):
            return
        if not self._single_consumer(add, add.output[0], mul.name):
            return
        if not self._single_consumer(mul_half, mul_half.output[0], mul.name):
            return

        gelu = make_node(
            "Gelu",
            inputs=[x],
            outputs=[mul.output[0]],
            name=f"{mul.name}/Gelu",
        )
        self += gelu
        self -= [div, erf, add, mul_half, mul]

    @staticmethod
    def _other_input(node: NodeProto, known: str) -> str | None:
        if len(node.input) != 2:
            return None
        if node.input[0] == known:
            return node.input[1]
        if node.input[1] == known:
            return node.input[0]
        return None

    @staticmethod
    def _is_close_scalar(value: np.ndarray | None, expected: float) -> bool:
        if value is None:
            return False
        arr = np.asarray(value)
        if arr.size != 1:
            return False
        return bool(
            np.isclose(float(arr.reshape(-1)[0]), expected, rtol=1e-5, atol=1e-7)
        )

    def _single_consumer(
        self, node: NodeProto, output_name: str, expected_consumer: str
    ) -> bool:
        consumers = self.get_output_node(node, output_name)
        return len(consumers) == 1 and consumers[0].name == expected_consumer
