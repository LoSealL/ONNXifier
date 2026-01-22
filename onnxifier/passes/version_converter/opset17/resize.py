"""
Copyright (C) 2024-2025 The ONNXIFIER Authors.

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

from typing import List, Literal, Optional

import numpy as np
from onnx.onnx_pb import NodeProto

from .... import OnnxGraph
from ...pattern import SingleNodePattern
from ...rewriter import Rewriter
from ...utils import make_constant
from . import OP_CONVERTER


@OP_CONVERTER.register("Resize", deps=["resize_move_size_to_scale"])
class Resize(Rewriter):
    """Remove new attributes in Resize operator.

    - antialias - INT (default 0)
    - axes - INTS
    - keep_aspect_ratio_policy - STRING (default "stretch")
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Resize"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        mode = self.get_attribute(node, "mode")
        antialias = self.get_attribute(node, "antialias")
        if antialias and mode in ("linear", "cubic"):
            raise ValueError(
                f"Resize({mode}) with antialiasing not supported below opset 18."
            )
        self.remove_attribute(node, "antialias")
        if ori_axes := self.get_attribute(node, "axes"):
            input_shape = graph.tensor_shape(node.input[0])
            input_rank = len(input_shape)
            axes = [i if i >= 0 else i + input_rank for i in ori_axes]  # type: ignore
            missing_axes = set(range(input_rank)) - set(axes)
            scales_node = self.get_input_node_or_die(node, 2)
            scales = self.get_value_or_die(scales_node)
            dtype = scales.dtype
            scales_value = {axis: scales[i] for i, axis in enumerate(axes)}
            scales_value.update({axis: 1.0 for axis in missing_axes})
            scales_value = [scales_value[axis] for axis in sorted(scales_value)]
            scales = np.array(scales_value, dtype=dtype)
            new_scales = make_constant(scales_node.name + "/new", scales)
            node.input[2] = new_scales.output[0]
            self -= scales_node
            self += new_scales
            keep_aspect_ratio_policy = self.get_attribute(
                node, "keep_aspect_ratio_policy"
            )
            keep_aspect_ratio_policy = f"{keep_aspect_ratio_policy}"  # force __repr__
            if keep_aspect_ratio_policy in ("not_larger", "not_smaller"):
                # rescale should use the original axes to specify the min/max range of
                # scales value.
                self._rescale(
                    node,
                    keep_aspect_ratio_policy,
                    axes,
                    scales,
                )
        self.remove_attribute(node, "axes")
        self.remove_attribute(node, "keep_aspect_ratio_policy")

    def _rescale(
        self,
        node: NodeProto,
        policy: Literal["not_larger", "not_smaller"],
        axes: Optional[List[int]],
        scales: np.ndarray,
    ):
        if axes is None:
            axes = list(range(len(scales)))
        axes = sorted(axes)
        scales = scales[axes]
        if policy == "not_larger":
            scales = scales.min()
        else:
            scales = scales.max()
        scales = np.tile(scales, [len(axes)])
        # append 1.0 for missing axes
        input_shape = self.graph.tensor_shape(node.input[0])
        new_scales = np.ones([len(input_shape)], dtype=scales.dtype)
        scales_iter = iter(scales)
        for i, _ in enumerate(input_shape):
            new_scales[i] = next(scales_iter) if i in axes else 1.0
        scales_cst = make_constant(f"{node.name}/new_scales", new_scales)
        node.input[2] = scales_cst.output[0]
        self += scales_cst
