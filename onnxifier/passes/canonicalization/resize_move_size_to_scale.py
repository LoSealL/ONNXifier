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

from typing import List, Sequence

import numpy as np
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(
    name="resize_move_size_to_scale",
    deps=["initializer_to_constant", "infer_shape", "shape_to_constant"],
)
class ResizeMoveSizeToScaleRewriter(Rewriter):
    """Move `size` input to `scale` input for Resize Op."""

    def __init__(self):
        super().__init__(SingleNodePattern("Resize"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node_pb = nodes[0]
        node = node_pb.name
        input_nodes = self.get_input_nodes(node_pb)
        if len(input_nodes) < 3:
            raise ValueError(
                f"Op '{node}' both scales and sizes are empty!"
                " Try fold_constant pass before this."
            )
        elif len(input_nodes) == 3:
            return
        _, roi, scales, sizes = input_nodes
        if sizes is None and scales is None:
            raise ValueError(
                f"Op '{node}' both scales and sizes are empty!"
                " Try fold_constant pass before this."
            )
        if scales is not None:
            return
        assert sizes is not None
        input_shape = graph.static_tensor_shape(node_pb.input[0])
        axes = self.get_attribute(node_pb, "axes") or range(len(input_shape))
        ct_mode = self.get_attribute(node_pb, "coordinate_transformation_mode")
        sizes_val = self.get_value_or_die(sizes)
        assert isinstance(axes, Sequence)
        axes = [int(i) for i in axes]  # type: ignore
        if roi is not None and ct_mode == "tf_crop_and_resize":
            roi_val = self.get_value_or_die(roi).reshape([2, -1])
            roi_size = []
            for i, j, k in zip(roi_val[0], roi_val[1], sizes_val):
                if i < 0:
                    i += k
                if j < 0:
                    j += k
                assert j >= i >= 0
                roi_size.append(j - i)
            scales_val = [sizes_val[i] / roi_size[i] for i, _ in enumerate(axes)]
        else:
            scales_val = [sizes_val[i] / input_shape[j] for i, j in enumerate(axes)]
        scales = make_constant(
            f"{node}/const/scales", np.array(scales_val, dtype="float32")
        )
        node_pb.input[2] = scales.output[0]
        node_pb.input.pop()  # remove `sizes`
        self += scales
        self -= sizes
