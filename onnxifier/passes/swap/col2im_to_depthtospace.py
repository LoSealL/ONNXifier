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
from onnx import NodeProto
from onnx.helper import make_node

from ... import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register("col2im_to_depthtospace")
class Col2ImToDepthToSpaceRewriter(Rewriter):
    """Convert simple Col2Im to DepthToSpace.

    Col2Im must satisfy strides == block_size

    Before:

        Col2Im

    After:

        Reshape -> DepthToSpace
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Col2Im"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]

        dilations = self.get_attribute(node, "dilations", [1])
        if isinstance(dilations, list) and any(i != 1 for i in dilations):
            return

        strides = self.get_attribute(node, "strides")
        if not isinstance(strides, list):
            return

        image_shape = self.get_value_or_die(node.input[1])
        block_size = self.get_value_or_die(node.input[2])
        input_shape = graph.tensor_shape(node.input[0])

        if (block_size != strides).any() or any(i != block_size[0] for i in block_size):
            return

        b, c, _ = input_shape
        output_shape = [b, c, *[i // s for i, s in zip(image_shape, strides)]]
        shape_cst = make_constant(
            f"{node.name}/shape", np.array(output_shape, dtype=np.int64)
        )
        reshape = make_node(
            "Reshape",
            [node.input[0], shape_cst.output[0]],
            [f"{node.name}/Reshape_output0"],
            name=f"{node.name}/Reshape",
        )
        d2s = make_node(
            "DepthToSpace",
            [reshape.output[0]],
            [node.output[0]],
            name=node.name,
            blocksize=int(block_size[0]),
            mode="CRD",
        )

        self += [shape_cst, reshape, d2s]
        self -= node
