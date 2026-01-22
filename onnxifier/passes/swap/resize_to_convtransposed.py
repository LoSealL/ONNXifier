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

from typing import List, Optional

import numpy as np
from onnx import NodeProto
from onnx.helper import make_node

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register("resize_to_convtransposed")
class ResizeToConvTransposedRewriter(Rewriter):
    """Convert eligible Resize to ConvTranspose with fixed weights.

    Conditions:
    - Input rank is 4 (NCHW)
    - Scales input exists and is constant/evaluable
    - Scale on N and C axes are 1
    - Spatial scales (H, W) are positive integers

    Result:
    - Replace with group-wise ConvTranspose (groups = C)
    - Weights are a constant of ones with shape (C, 1, sH, sW)
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Resize"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        input_shape = graph.tensor_shape(node.input[0])
        if not self._is_valid_input_shape(input_shape):
            return
        channels = int(input_shape[1])
        spatial_size = list(map(int, input_shape[2:]))

        scales = self._get_scales(node, spatial_size)
        if scales is None or np.all(scales == 1):
            return

        mode = self.get_attribute(node, "mode")
        built = None
        if mode is None or mode == "nearest":
            built = self._build_nearest(node, channels, scales)
        elif mode == "linear":
            built = self._build_linear(node, channels, scales)
        if built is None:
            return

        self += built
        self -= node

    def _is_valid_input_shape(self, input_shape):
        valid = input_shape is not None
        valid &= len(input_shape) >= 3
        valid &= isinstance(input_shape[1], int)
        valid &= input_shape[1] > 0
        valid &= all(isinstance(d, int) for d in input_shape[2:])
        return valid

    def _get_scales(
        self, node: NodeProto, input_shape: List[int]
    ) -> Optional[List[int]]:
        # Prefer explicit scales if present; otherwise, derive from sizes
        scales = None
        axes: list[int] = self.get_attribute(node, "axes", [])  # type: ignore
        rank = len(input_shape) + 2
        axes = [ax + rank if ax < 0 else ax for ax in axes]
        if len(node.input) >= 3 and node.input[2]:
            scales = self.get_value(node.input[2])
            if scales is not None:
                scales = np.asarray(scales)
                if axes:
                    full_scales = np.ones(rank - 2, dtype=np.float32)
                    for i, ax in enumerate(axes):
                        if ax < 2:
                            if scales[i] != 1.0:
                                return None
                            continue
                        full_scales[ax - 2] = float(scales[i])
                else:
                    full_scales = scales[2:]
                scales = full_scales
        if scales is None and len(node.input) >= 4 and node.input[3]:
            sizes = self.get_value(node.input[3])
            if sizes is None:
                return None
            sizes = np.asarray(sizes)
            # Build full sizes per rank
            if not axes:
                full_sizes = sizes[2:].astype(np.int64)
            else:
                full_sizes = np.empty(rank - 2, dtype=np.int64)
                for i, ax in enumerate(axes):
                    if ax < 2:
                        continue
                    # ensure positive
                    full_sizes[ax - 2] = int(sizes[i])
            scales = full_sizes / np.asarray(input_shape, dtype=np.float32)
        if scales is not None:
            if np.any(scales != scales.astype(np.int64)):
                return None
            return scales.astype(np.int64).tolist()

    def _build_nearest(self, node: NodeProto, channels: int, scales: List[int]):
        weight_value = np.ones((channels, 1, *scales), dtype=np.float32)
        weight_cst = make_constant(f"{node.name}/weights", weight_value)
        convt = make_node(
            "ConvTranspose",
            inputs=[node.input[0], weight_cst.output[0]],
            outputs=list(node.output),
            name=node.name,
            strides=scales,
            group=channels,
        )
        return weight_cst, convt

    def _build_linear(self, node: NodeProto, channels: int, scales: List[int]):
        ctm = self.get_attribute(node, "coordinate_transformation_mode", "half_pixel")
        # Default to half_pixel-like mapping
        kernel_sizes = [2 * s for s in scales]
        pads = [(s + 1) // 2 for s in scales]
        pads = np.repeat(pads, 2)
        output_pads = [s % 2 for s in scales]

        # Generate appropriate kernel based on coordinate transformation mode
        if ctm == "asymmetric":
            pads = np.repeat([s - 1 for s in scales], 2)
            output_pads = [s - 1 for s in scales]
            base = self._asymmetric_kernel(*kernel_sizes)
        elif ctm in ("align_corners", "tf_crop_and_resize"):
            return None
        else:  # half_pixel
            base = self._half_pixel_kernel(*kernel_sizes)

        weight_value = np.tile(base.T[..., None, None], [1, channels]).T
        weight_cst = make_constant(f"{node.name}/weights", weight_value)
        convt_kwargs: dict = dict(
            strides=scales,
            group=channels,
            pads=pads,
        )
        if output_pads:
            convt_kwargs["output_padding"] = output_pads
        convt = make_node(
            "ConvTranspose",
            inputs=[node.input[0], weight_cst.output[0]],
            outputs=list(node.output),
            name=node.name,
            **convt_kwargs,
        )
        return weight_cst, convt

    def _half_pixel_kernel(self, *kernel_size: int) -> np.ndarray:
        """Generate kernel for half_pixel coordinate transformation mode.

        For half pixel mode:

        ..math::

            x_original = (x_resized + 0.5) / scale - 0.5
        """
        scales = [(k + 1) // 2 for k in kernel_size]
        kernels = np.array(
            [[(i + 0.5 - (s + 1) // 2) / s - 0.5 for i in range(s * 2)] for s in scales]
        )
        kernels = 1 - np.abs(kernels)
        kernels = np.clip(kernels, 0, 1)
        kernel = kernels[0]
        for d, k in enumerate(kernels[1:]):
            k = np.expand_dims(k, axis=list(-np.arange(1, d + 2)))
            kernel = kernel[..., None] @ k.T
        return kernel.astype(np.float32)

    def _asymmetric_kernel(self, *kernel_size: int) -> np.ndarray:
        """Generate kernel for asymmetric coordinate transformation mode.

        For asymmetric mode:

        ..math::

            x_original = x_resized / scale
        """
        scales = [(k + 1) // 2 for k in kernel_size]
        kernels = []
        for s, k in zip(scales, kernel_size):
            kernels.append(np.array([1 - np.abs(s - i) / s for i in range(1, k)]))
        kernels = np.clip(kernels, 0, 1)
        kernel = kernels[0]
        for _, ker in enumerate(kernels[1:]):
            kernel = kernel[..., None] @ ker[None]
        return kernel.astype(np.float32)
