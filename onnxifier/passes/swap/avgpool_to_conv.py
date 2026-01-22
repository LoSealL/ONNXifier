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

from typing import List, cast

import numpy as np
from onnx import NodeProto
from onnx.helper import make_node

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register("avgpool_to_conv")
class AvgPoolToConvRewriter(Rewriter):
    """Replace eligible AveragePool with group-wise Conv.

    Conditions for conversion:
    - Operator is `AveragePool` (not `GlobalAveragePool`).
    - `ceil_mode` is 0 or not set.
    - If `count_include_pad` is 0, then all pads must be 0.
      If `count_include_pad` is 1, arbitrary pads are supported.

    Result:
    - Replace with `Conv` with group = C and weights filled with 1/(prod(kernel))
      of shape (C, 1, *kernel_shape). Strides/pads/auto_pad are preserved.
    """

    def __init__(self):
        super().__init__(SingleNodePattern("AveragePool"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]

        # Extract attributes
        kernel_shape_attr = self.get_attribute(node, "kernel_shape")
        if (
            not isinstance(kernel_shape_attr, list)
            or len(kernel_shape_attr) == 0
            or not all(isinstance(k, (int, np.integer)) for k in kernel_shape_attr)
        ):
            return
        kernel_shape = [int(k) for k in kernel_shape_attr]  # type: ignore
        spatial_rank = len(kernel_shape)

        strides_attr = self.get_attribute(node, "strides", None)
        if isinstance(strides_attr, list) and len(strides_attr) == spatial_rank:
            validated_strides: List[int] = []
            for s in strides_attr:
                if isinstance(s, (int, np.integer)):
                    validated_strides.append(int(cast(int, s)))
                else:
                    validated_strides = []
                    break
            if validated_strides:
                strides = validated_strides
            else:
                strides = [1] * spatial_rank
        else:
            strides = [1] * spatial_rank

        pads_attr = self.get_attribute(node, "pads", None)
        if isinstance(pads_attr, list) and len(pads_attr) == 2 * spatial_rank:
            validated_pads: List[int] = []
            for p in pads_attr:
                if isinstance(p, (int, np.integer)):
                    validated_pads.append(int(cast(int, p)))
                else:
                    validated_pads = []
                    break
            if validated_pads:
                pads = validated_pads
            else:
                pads = [0] * (2 * spatial_rank)
        else:
            pads = [0] * (2 * spatial_rank)

        auto_pad_attr = self.get_attribute(node, "auto_pad", "NOTSET")
        auto_pad = str(auto_pad_attr) if auto_pad_attr is not None else "NOTSET"

        ceil_mode_attr = self.get_attribute(node, "ceil_mode", 0)
        ceil_mode = (
            int(ceil_mode_attr) if isinstance(ceil_mode_attr, (int, np.integer)) else 0
        )

        cip_attr = self.get_attribute(node, "count_include_pad", 0)
        count_include_pad = (
            int(cip_attr) if isinstance(cip_attr, (int, np.integer)) else 0
        )

        # Unsupported settings
        if ceil_mode != 0:
            return
        if count_include_pad == 0 and any(int(p) != 0 for p in pads):
            # Excluding padded zeros cannot be modeled with a single fixed Conv
            return

        # Validate input shape and channels
        input_shape = graph.tensor_shape(node.input[0])
        if input_shape is None or len(input_shape) < spatial_rank + 2:
            return
        channels_any = input_shape[1]
        if not isinstance(channels_any, int) or channels_any <= 0:
            return
        channels: int = int(channels_any)

        # Build depthwise weights: (C, 1, *kernel) with average factor
        kernel_prod = int(np.prod([int(k) for k in kernel_shape]))
        scale = 1.0 / float(kernel_prod)
        weight_value = np.full(
            (channels, 1, *[int(k) for k in kernel_shape]), scale, dtype=np.float32
        )
        weight_cst = make_constant(f"{node.name}/weights", weight_value)

        conv_kwargs: dict = dict(strides=[int(s) for s in strides], group=channels)
        if auto_pad != "NOTSET":
            conv_kwargs["auto_pad"] = auto_pad
        else:
            conv_kwargs["pads"] = [int(p) for p in pads]

        conv = make_node(
            "Conv",
            inputs=[node.input[0], weight_cst.output[0]],
            outputs=list(node.output),
            name=node.name,
            **conv_kwargs,
        )

        self += [weight_cst, conv]
        self -= node
