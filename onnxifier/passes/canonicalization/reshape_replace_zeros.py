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

# pylint: disable=arguments-differ

from typing import List

import numpy as np
from onnx.onnx_pb import NodeProto

from ...graph import OnnxGraph
from .. import L2
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@L2.register("reshape_replace_zeros", deps=["infer_shape"])
class ReshapeReplaceZerosRewriter(Rewriter):
    """Replace zeros in shape data with static values."""

    def __init__(self):
        super().__init__(SingleNodePattern("Reshape"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        output_shape = graph.tensor_shape(nodes[0].output[0])
        target_shape = self.get_value(nodes[0].input[1])
        if target_shape is not None:
            if not any(i == 0 for i in target_shape):
                return
            assert len(target_shape) == len(output_shape)
            shape_changed = False
            target_shape = list(target_shape)
            for k, (i, j) in enumerate(zip(target_shape, output_shape)):
                if i in (0, -1) and isinstance(j, int):
                    target_shape[k] = j
                    shape_changed = True
        elif all(isinstance(i, int) for i in output_shape):
            # In case converted from OpenVINO and somehow the shape can't be evaluated
            target_shape = output_shape
            shape_changed = True
        else:
            return
        if shape_changed:
            shape = make_constant(
                f"{nodes[0].name}/shape", np.array(target_shape, dtype=np.int64)
            )
            nodes[0].input[1] = shape.output[0]
            self += shape
