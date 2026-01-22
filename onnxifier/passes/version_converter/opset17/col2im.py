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
from onnx.helper import make_node, make_tensor

from .... import OnnxGraph
from ...pattern import SingleNodePattern
from ...rewriter import Rewriter
from ...utils import make_constant
from . import OP_CONVERTER


@OP_CONVERTER.register("Col2Im")
class Col2Im(Rewriter):
    """Convert Col2Im v18 to equivalent counterpart using v17 supported operations.

    Col2Im rearranges column blocks back into a multidimensional image.
    Since Col2Im doesn't exist in opset 17, we implement it using supported operations:
    Reshape, Transpose, and other basic tensor operations.
    """

    def __init__(self):
        super().__init__(SingleNodePattern("Col2Im"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        dilations = self.get_attribute(node, "dilations")
        if isinstance(dilations, list) and any(i != 1 for i in dilations):
            raise ValueError("Dilation is not supported in Col2Im conversion.")

        # Get block_shape from input to determine spatial dimensions
        block_shape = self.get_value_or_die(node.input[2])
        spatial_rank = len(block_shape)

        stride = self.get_attribute(node, "strides")
        padding = self.get_attribute(node, "pads")
        if stride is None:
            stride = [1] * spatial_rank
        if padding is None:
            padding = [0] * (2 * spatial_rank)
        assert isinstance(stride, list) and isinstance(padding, list)
        stride = [int(s) for s in stride if isinstance(s, (int, float))]
        padding = [int(p) for p in padding if isinstance(p, (int, float))]

        # Try the simple case first: use DepthToSpace if possible
        if self._try_depth_to_space_conversion(graph, node, stride, padding):
            return

        # Fall back to general implementation using basic operations
        self._general_col2im_implementation(graph, node, stride, padding)

    def _try_depth_to_space_conversion(
        self,
        graph: OnnxGraph,
        node: NodeProto,
        stride: List[int],
        padding: List[int],
    ) -> bool:
        """Try to convert Col2Im to DepthToSpace for simple cases."""

        output_shape = graph.tensor_shape(node.output[0])[2:]
        if all(isinstance(i, int) for i in output_shape):
            image_shape = [int(i) for i in output_shape]
        else:
            image_shape = self.get_value_or_die(node.input[1])
        block_shape = self.get_value_or_die(node.input[2])
        if not all(block_shape == stride):
            return False
        if any(i != block_shape[0] for i in block_shape):
            return False
        if any(i != 0 for i in padding):
            return False
        input_shape = graph.tensor_shape(node.input[0])
        # Simple case: can use DepthToSpace
        b, c, _ = input_shape
        ds_shape = [b, c, *[i // s for i, s in zip(image_shape, stride)]]
        shape_cst = make_constant(
            f"{node.name}/shape", np.array(ds_shape, dtype=np.int64)
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
            blocksize=int(block_shape[0]),
            mode="CRD",
        )

        self += [shape_cst, reshape, d2s]
        self -= node
        return True

    def _general_col2im_implementation(
        self,
        graph: OnnxGraph,
        node: NodeProto,
        stride: List[int],
        padding: List[int],
    ):
        """General Col2Im implementation using ScatterElements for complex cases.

        Col2Im rearranges column data back to image format.
        Implementation: Col2Im(x) = Crop(ScatterElements(z, indices, x))
        where z is zero tensor with padded output shape.
        """

        # Get input dimensions and parameters
        input_shape = graph.static_tensor_shape(node.input[0])  # [B, C*kH*kW, L]
        block_shape = self.get_value_or_die(node.input[2])  # [kH, kW]
        image_shape = self.get_value_or_die(node.input[1])  # [H, W]
        dtype = graph.tensor_type(node.output[0])

        # Convert to integers, handling dynamic shapes if needed
        batch = int(input_shape[0])
        channels_times_kernel = int(input_shape[1])
        L = int(input_shape[2])

        kernel_size = int(np.prod(block_shape))
        channels = channels_times_kernel // kernel_size

        # Calculate padded dimensions
        paddings = np.reshape(padding, (2, -1)).T
        image_pad_shape = [i + p[0] + p[1] for i, p in zip(image_shape, paddings)]

        # Step 1: Reshape input from [B, C*kH*kW, L] to [B, C, kH*kW, L]
        update_shape = make_constant(
            f"{node.name}/update_shape",
            np.array([batch, channels, kernel_size * L], dtype=np.int64),
        )
        update_reshape = make_node(
            "Reshape",
            [node.input[0], update_shape.output[0]],
            [f"{node.name}/update_reshape_output0"],
            name=f"{node.name}/update_reshape",
        )

        # Step 2: Create zero tensor z with padded shape [B, C, padded_H*padded_W]
        flatten_shape = make_constant(
            f"{node.name}/flatten_shape",
            np.array([batch, channels, np.prod(image_pad_shape)], dtype=np.int64),
        )

        zero_tensor = make_node(
            "ConstantOfShape",
            [flatten_shape.output[0]],
            [f"{node.name}/zero_tensor"],
            name=f"{node.name}/zero_tensor",
            value=make_tensor("value", dtype, [1], [0]),
        )

        # Step 3: Calculate ScatterElements indices
        # We need to map each position of input block [kH, kW, L]
        # to the correct position in output tensor [H, W]
        # and it is repeated on batch size and channel.

        # Calculate how many patches fit in each dimension of the output
        spatial_shape = [
            int(np.ceil((i + p[0] + p[1] - k + 1) / s))
            for i, p, k, s in zip(image_shape, paddings, block_shape, stride)
        ]
        assert L == np.prod(spatial_shape)
        # Input X (omit batch and channel) has shape [*block_shape, *spatial_shape]
        block_indices = np.meshgrid(*[np.arange(d) for d in block_shape], indexing="ij")
        block_indices = np.stack(block_indices, axis=-1)
        block_indices = block_indices.reshape(-1, len(block_shape))
        spatial_indices = np.meshgrid(
            *[np.arange(d) for d in spatial_shape], indexing="ij"
        )
        spatial_indices = np.stack(spatial_indices, axis=-1)
        spatial_indices = spatial_indices.reshape(-1, len(spatial_shape))
        # Create all combinations of kernel and spatial indices
        kdx = np.repeat(block_indices, len(spatial_indices), axis=0)
        sdx = np.tile(spatial_indices, (len(block_indices), 1))
        # Calculate output positions using correct Col2Im formula
        output_positions = []
        for i in range(len(spatial_shape)):
            pos = sdx[:, i] * stride[i] + kdx[:, i]  # + paddings[i, 0]
            output_positions.append(pos)
        odx = np.stack(output_positions, axis=1)

        # Create indices array [kernel_size * L]
        indices_array = np.ravel_multi_index(odx.T, image_pad_shape)  # type: ignore
        indices_array = np.tile(indices_array, [batch, channels, 1])
        indices_array = indices_array.reshape([batch, channels, -1])

        indices_const = make_constant(
            f"{node.name}/indices",
            indices_array,
        )

        # Step 4: Apply ScatterElements to place patches
        # ScatterElements: scatter updates into zero tensor based on indices
        scatter = make_node(
            "ScatterElements",
            [zero_tensor.output[0], indices_const.output[0], update_reshape.output[0]],
            [f"{node.name}/scattered"],
            name=f"{node.name}/scatter",
            axis=2,  # Scatter along spatial dimension,
            reduction="add",
        )

        # Step 5: Reshape back to spatial dimensions [B, C, padded_H, padded_W]
        padded_shape = make_constant(
            f"{node.name}/padded_shape",
            np.array([batch, channels, *image_pad_shape], dtype=np.int64),
        )
        reshape_back = make_node(
            "Reshape",
            [scatter.output[0], padded_shape.output[0]],
            [f"{node.name}/padded_result"],
            name=f"{node.name}/reshape_back",
        )

        # Step 6: Crop padding if needed
        if any(p > 0 for p in padding):
            starts = make_constant(
                f"{node.name}/crop_starts",
                np.concatenate([[0, 0], paddings[:, 0]]).astype(np.int64),
            )
            image_size = [i + p for i, p in zip(image_shape, paddings[:, 0])]
            ends = make_constant(
                f"{node.name}/crop_ends",
                np.concatenate([[batch, channels], image_size]).astype(np.int64),
            )
            crop = make_node(
                "Slice",
                [reshape_back.output[0], starts.output[0], ends.output[0]],
                [node.output[0]],
                name=node.name,
            )

            nodes_to_add = [
                update_shape,
                update_reshape,
                flatten_shape,
                indices_const,
                padded_shape,
                starts,
                ends,
                zero_tensor,
                scatter,
                reshape_back,
                crop,
            ]
        else:
            # No padding needed, use result directly
            final_rename = make_node(
                "Identity",
                [reshape_back.output[0]],
                [node.output[0]],
                name=node.name,
            )

            nodes_to_add = [
                update_shape,
                update_reshape,
                flatten_shape,
                indices_const,
                padded_shape,
                zero_tensor,
                scatter,
                reshape_back,
                final_rename,
            ]

        self += nodes_to_add
        self -= node
