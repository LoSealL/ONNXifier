"""
Copyright (C) 2025 The ONNXIFIER Authors.

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

from typing import List

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant

ACT_SHAVE = (
    "Sigmoid",
    "Swish",
    "Mish",
    "Gelu",
    "Tanh",
)


@PASSES.register(
    "insert_conv_before_act_shave",
    deps=["fuse_mish", "fuse_swish"],
)
class InsertConvBeforeActShaveRewriter(Rewriter):
    def __init__(self):
        pattern = GraphPattern()
        patterns = [SingleNodePattern(act) for act in ACT_SHAVE]
        # TODO: quantized?
        pattern.add_edge(sum(patterns), SingleNodePattern("Conv"))
        super().__init__(pattern=pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        act_node, conv_node = nodes
        if len(graph.onnx_successors(act_node)) > 1:
            return  # make sure act has only one conv user

        _, channels, _, _ = graph.tensor_shape(act_node.output[0])
        assert isinstance(channels, int) and channels > 1
        weight1 = make_constant(
            f"{act_node.name}/conv1/weight",
            np.ones((channels, 1, 1, 1), np.float32),
        )
        conv1 = make_node(
            "Conv",
            [act_node.output[0], weight1.output[0]],
            [f"{act_node.name}/conv1_output0"],
            name=f"{act_node.name}/conv1",
            group=channels,
        )
        conv_node.input[0] = conv1.output[0]
        self += [weight1, conv1]
