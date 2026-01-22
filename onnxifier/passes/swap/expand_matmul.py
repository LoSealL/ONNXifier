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

# pylint: disable=arguments-differ

from typing import List

import numpy as np
from onnx.helper import make_node

from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(name="expand_matmul", deps=["initializer_to_constant", "infer_shape"])
class ExpandMatMulRewriter(Rewriter):
    """Expand MatMul's input and output shape to factor aligned size"""

    def __init__(self, factor: int = 2):
        """
        Args:
            factor (int): The alignment factor for expanding dimensions
        """
        super().__init__(pattern=SingleNodePattern("MatMul"))
        self.factor = factor

    def _align_to_factor(self, size: int) -> int:
        """Align size to the next multiple of factor"""
        return ((size + self.factor - 1) // self.factor) * self.factor

    def _expand_shape_skip_batch(self, shape: List[int]) -> List[int]:
        """Expand shape but skip batch dimensions for MatMul

        For MatMul:
        - 2D: [M, K] -> all dimensions can be expanded
        - 3D+: [B..., M, K] -> only last 2 dimensions (M, K) can be expanded
        """
        if len(shape) <= 2:
            # 2D or less: expand all dimensions
            return [self._align_to_factor(dim) for dim in shape]
        else:
            # 3D+: keep batch dimensions unchanged, expand only last 2 dimensions
            batch_dims = shape[:-2]  # Keep batch dimensions as-is
            matrix_dims = [
                self._align_to_factor(dim) for dim in shape[-2:]
            ]  # Expand matrix dimensions
            return batch_dims + matrix_dims

    def rewrite(self, graph: OnnxGraph, nodes: List):
        matmul_node = nodes[0]

        # Get input shapes
        input_a_shape_raw = graph.tensor_shape(matmul_node.input[0])
        input_b_shape_raw = graph.tensor_shape(matmul_node.input[1])

        # Convert to int, skip if any dimension is dynamic (string)
        try:
            input_a_shape = [int(dim) for dim in input_a_shape_raw]
            input_b_shape = [int(dim) for dim in input_b_shape_raw]
        except (ValueError, TypeError):
            # Skip if any dimension is dynamic
            return

        # Check if we need to expand any dimensions (skip batch dimensions)
        expanded_a_shape = self._expand_shape_skip_batch(input_a_shape)
        expanded_b_shape = self._expand_shape_skip_batch(input_b_shape)

        # If no expansion needed, skip
        if input_a_shape == expanded_a_shape and input_b_shape == expanded_b_shape:
            return

        # Handle input A expansion
        new_input_a_name = self._expand_tensor(
            graph, matmul_node, 0, input_a_shape, expanded_a_shape, "A"
        )
        if new_input_a_name is None:
            return  # Skip if we can't expand input A

        # Handle input B expansion
        new_input_b_name = self._expand_tensor(
            graph, matmul_node, 1, input_b_shape, expanded_b_shape, "B"
        )
        if new_input_b_name is None:
            return  # Skip if we can't expand input B

        # Calculate output shape after expansion
        expanded_output_shape = self._calculate_matmul_output_shape(
            expanded_a_shape, expanded_b_shape
        )
        original_output_shape = self._calculate_matmul_output_shape(
            input_a_shape, input_b_shape
        )

        # Create new MatMul node with expanded inputs
        expanded_matmul_name = f"{matmul_node.name}_expanded"
        expanded_output_name = f"{matmul_node.output[0]}_expanded"

        expanded_matmul = make_node(
            op_type="MatMul",
            inputs=[new_input_a_name, new_input_b_name],
            outputs=[expanded_output_name],
            name=expanded_matmul_name,
        )

        # Add slice node to restore original output shape
        slice_node = self._create_slice_node(
            expanded_output_name,
            matmul_node.output[0],
            expanded_output_shape,
            original_output_shape,
            f"{matmul_node.name}_slice",
        )

        # Replace the original MatMul
        self -= matmul_node
        self += [expanded_matmul, slice_node]

    def _expand_tensor(
        self,
        graph: OnnxGraph,
        matmul_node,
        input_index: int,
        original_shape: List[int],
        expanded_shape: List[int],
        suffix: str,
    ) -> str:
        """Unified method to expand tensor (either input or weight)"""
        input_name = matmul_node.input[input_index]

        # If no expansion needed, return original name
        if original_shape == expanded_shape:
            return input_name

        # Check if this is a constant/weight node
        input_node = self.get_input_node(matmul_node, input_index)
        if input_node and input_node.op_type == "Constant":
            # Handle as weight/constant
            return self._expand_weight(input_node, original_shape, expanded_shape)
        else:
            # Handle as graph input
            return self._expand_input(
                graph, input_name, original_shape, expanded_shape, suffix
            )

    def _expand_input(
        self,
        graph: OnnxGraph,
        input_name: str,
        original_shape: List[int],
        expanded_shape: List[int],
        suffix: str,
    ) -> str:
        """Expand input using Pad operator to maintain graph input/output shapes"""
        # Calculate padding needed for each dimension
        # ONNX Pad format: [begin_0, begin_1, ..., end_0, end_1, ...]
        pads_begin = []
        pads_end = []
        for orig_dim, exp_dim in zip(original_shape, expanded_shape):
            pad_size = exp_dim - orig_dim
            pads_begin.append(0)  # No padding at the beginning
            pads_end.append(pad_size)  # Pad at the end

        pads = pads_begin + pads_end

        # If no padding needed, return original name
        if all(pad == 0 for pad in pads):
            return input_name

        # Create constant node for pads
        pads_array = np.array(pads, dtype=np.int64)
        pads_node = make_constant(f"{input_name}_pads_{suffix}", pads_array)

        # Create Pad node
        padded_name = f"{input_name}_padded_{suffix}"
        pad_node = make_node(
            op_type="Pad",
            inputs=[input_name, pads_node.output[0]],
            outputs=[padded_name],
            name=f"{input_name}_pad_{suffix}",
            mode="constant",  # Zero padding
        )

        # Add nodes to graph
        self += [pads_node, pad_node]

        return padded_name

    def _expand_weight(
        self, weight_node, original_shape: List[int], expanded_shape: List[int]
    ) -> str:
        """Expand weight tensor by zero-padding"""
        weight_value = self.get_value_or_die(weight_node)

        # Create expanded weight with zero padding
        expanded_weight_value = np.zeros(expanded_shape, dtype=weight_value.dtype)

        # Copy original values to the expanded tensor
        slices = tuple(slice(0, dim) for dim in original_shape)
        expanded_weight_value[slices] = weight_value

        # Create new constant node with expanded weight
        expanded_weight_node = make_constant(
            weight_node.name + "_expand", expanded_weight_value
        )

        # Remove old weight node and add new one
        self -= weight_node
        self += expanded_weight_node

        return expanded_weight_node.output[0]

    def _calculate_matmul_output_shape(
        self, shape_a: List[int], shape_b: List[int]
    ) -> List[int]:
        """Calculate MatMul output shape"""
        # For MatMul: (..., K) x (K, N) -> (..., N)
        # or (..., M, K) x (..., K, N) -> (..., M, N) for batch dimensions

        if len(shape_a) == 1 and len(shape_b) == 1:
            # Vector dot product: (K,) x (K,) -> scalar
            return []
        elif len(shape_a) == 1:
            # (K,) x (..., K, N) -> (..., N)
            return shape_b[:-2] + [shape_b[-1]]
        elif len(shape_b) == 1:
            # (..., M, K) x (K,) -> (..., M)
            return shape_a[:-1]
        else:
            # (..., M, K) x (..., K, N) -> (..., M, N)
            # Broadcast batch dimensions
            batch_a = shape_a[:-2]
            batch_b = shape_b[:-2]

            # Simple broadcast (assuming compatible shapes)
            max_batch_len = max(len(batch_a), len(batch_b))
            result_batch = []

            for i in range(max_batch_len):
                dim_a = (
                    batch_a[i - max_batch_len]
                    if i - max_batch_len >= -len(batch_a)
                    else 1
                )
                dim_b = (
                    batch_b[i - max_batch_len]
                    if i - max_batch_len >= -len(batch_b)
                    else 1
                )
                result_batch.append(max(dim_a, dim_b))

            return result_batch + [shape_a[-2], shape_b[-1]]

    def _create_slice_node(
        self,
        input_name: str,
        output_name: str,
        input_shape: List[int],
        output_shape: List[int],
        node_name: str,
    ):
        """Create slice node to crop expanded output back to original shape"""

        # Create starts (all zeros)
        starts = np.zeros(len(input_shape), dtype=np.int64)
        starts_node = make_constant(f"{node_name}_starts", starts)

        # Create ends (original shape dimensions)
        ends = np.array(output_shape, dtype=np.int64)
        ends_node = make_constant(f"{node_name}_ends", ends)

        # Create axes (all dimensions)
        axes = np.arange(len(input_shape), dtype=np.int64)
        axes_node = make_constant(f"{node_name}_axes", axes)

        # Create slice node
        slice_node = make_node(
            op_type="Slice",
            inputs=[
                input_name,
                starts_node.output[0],
                ends_node.output[0],
                axes_node.output[0],
            ],
            outputs=[output_name],
            name=node_name,
        )

        # Add constant nodes
        self += [starts_node, ends_node, axes_node]

        return slice_node
