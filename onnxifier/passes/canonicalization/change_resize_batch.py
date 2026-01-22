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

from typing import List

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(
    name="change_resize_batch",
    deps=["resize_move_size_to_scale", "infer_shape"],
)
class ChangeResizeBatch(Rewriter):
    """Change Resize's batch size to 1, which is only for LNL"""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Resize"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        resize = nodes[0]
        i_shape = graph.tensor_shape(resize.input[0])
        o_shape = graph.tensor_shape(resize.output[0])
        if len(i_shape) != 4 or len(o_shape) != 4 or i_shape[0] == 1:
            return

        # attributes
        default_attributes = {
            "antialias": 0,
            "axes": None,
            "coordinate_transformation_mode": "half_pixel",
            "cubic_coeff_a": -0.75,
            "exclude_outside": 1,
            "extrapolation_value": 0.0,
            "keep_aspect_ratio_policy": "stretch",
            "mode": "nearest",
            "nearest_mode": "round_prefer_floor",
        }
        attributes = {}
        for name in default_attributes:
            attributes[name] = self.get_attribute(resize, name)

        # input reshape
        N, C, H, W = i_shape
        i_cst = make_constant(
            name=f"{resize.name}/InputShapeCst",
            value=np.array([1, -1, H, W], dtype="int64"),
        )
        i_reshape = make_node(
            op_type="Reshape",
            inputs=[resize.input[0], i_cst.output[0]],
            outputs=[f"{resize.name}/input_reshape_output"],
            name=f"{resize.name}/InputReshape",
        )

        # new resize, it depends on resize_move_size_to_scale
        input_nodes = self.get_input_nodes(resize)
        if len(input_nodes) != 3:
            return
        new_resize = make_node(
            op_type="Resize",
            inputs=[i_reshape.output[0]] + resize.input[1:],
            outputs=[f"{resize.name}_Resize_output"],
            name=f"{resize.name}/Resize",
            **attributes,
        )

        # output reshape
        o_cst = make_constant(
            name=f"{resize.name}/OnputShapeCst", value=np.array(o_shape, dtype="int64")
        )
        o_reshape = make_node(
            op_type="Reshape",
            inputs=[new_resize.output[0], o_cst.output[0]],
            outputs=resize.output[:],
            name=f"{resize.name}/OutputReshape",
        )
        self += [i_cst, i_reshape, new_resize, o_cst, o_reshape]
        self -= [resize]
