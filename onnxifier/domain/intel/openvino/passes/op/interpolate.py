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

from contextlib import suppress

import numpy as np
import onnx
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from .. import OnnxGraph, cast_in, make_constant
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="interpolate")
class Interpolate(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/image/interpolate-11.html

    https://onnx.ai/onnx/operators/onnx__Resize.html
    """

    def _to_mode(self, mode):
        if mode == "linear_onnx":
            return "linear"
        return mode

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        shape_calculation_mode = self.get_attribute(ori_node, "shape_calculation_mode")
        if shape_calculation_mode is None:
            # for opset1 compatibility
            return self.replace_opset1(graph, ori_node)
        # axes
        version = self.get_attribute(ori_node, "version")
        if len(ori_node.input) > 3:
            axes = self.get_value(ori_node.input[3])
            ori_node.input.pop(3)
        elif version == "opset11":
            # axes is optional
            if len(ori_node.input) > 2:
                axes = self.get_value(ori_node.input[2])
            else:
                image_shape = graph.tensor_shape(ori_node.input[0])
                axes = np.arange(len(image_shape))
        else:
            rank = graph.tensor_shape(ori_node.input[1])[0]
            if isinstance(rank, str) or rank < 1:
                raise ValueError(f"Input shape[0] is dynamic: {rank}")
            # NOTE: 1. negative axis causes a bug in onnxruntime
            # NOTE: 2. axis less than input rank causes a bug in openvino
            # Solution to #1: convert to positive axis (here line:64)
            # Solution to #2: use pass "resize_remove_axes"
            axes = np.arange(-rank, 0) + len(graph.tensor_shape(ori_node.input[0]))
        self.set_attribute(ori_node, "axes", axes)
        while len(ori_node.input) < 3:
            ori_node.input.append("")
        if shape_calculation_mode == "sizes":
            ori_node.input.append(ori_node.input[1])
            ori_node.input[1] = ori_node.input[2] = ""
            size_type = graph.tensor_type(ori_node.input[3])
            if size_type != onnx.TensorProto.INT64:
                self += cast_in(ori_node, 3, onnx.TensorProto.INT64)
        elif shape_calculation_mode == "scales":
            if version == "opset11":
                ori_node.input[2] = ori_node.input[1]
            ori_node.input[1] = ""

        mode = self.get_attribute(ori_node, "mode") or "nearest"
        ctm = self.get_attribute(ori_node, "coordinate_transformation_mode")
        nearest_mode = self.get_attribute(ori_node, "nearest_mode")
        cubic_coeff_a = self.get_attribute(ori_node, "cubic_coeff")
        pads_begin = self.get_attribute(ori_node, "pads_begin")
        pads_begin = list(map(int, pads_begin.split(",")))  # type: ignore
        pads_end = self.get_attribute(ori_node, "pads_end")
        pads_end = list(map(int, pads_end.split(",")))  # type: ignore

        if any(pads_begin) or any(pads_end):
            raise ValueError("Interpolate does not support converting with padding")

        return make_node(
            "Resize",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            mode=self._to_mode(mode),
            coordinate_transformation_mode=ctm or "half_pixel",
            nearest_mode=nearest_mode or "round_prefer_floor",
            cubic_coeff_a=cubic_coeff_a or -0.75,
            axes=axes,
        )

    def replace_opset1(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        """Interpolate opset1 doesn't have attribute "shape_calculation_mode", we
        have to guess the scales_or_size based on its data type.
        """
        # get scales_or_size
        scales_or_size = self.get_value_or_die(ori_node.input[1])
        # size to scales
        input_dims = np.array(graph.tensor_shape(ori_node.input[0]))
        scales = scales_or_size
        with suppress(ValueError):
            np.iinfo(scales_or_size.dtype)  # integers for size
            if scales.size < input_dims.size:
                scales = np.concatenate(
                    [input_dims[: input_dims.size - scales.size], scales]
                )
            scales = np.true_divide(scales, input_dims).astype("float32")
        if scales.size < input_dims.size:
            preleading_ones = np.ones_like(input_dims[: input_dims.size - scales.size])
            scales = np.concatenate([preleading_ones, scales]).astype("float32")
        # make empty roi
        ori_node.input[1] = ""
        # make scales
        scale_node = make_constant(f"{ori_node.name}/scales", scales)
        if len(ori_node.input) > 2:
            ori_node.input[2] = scale_node.output[0]
        else:
            ori_node.input.append(scale_node.output[0])
        self += scale_node
        # axes
        if len(ori_node.input) > 3:
            axes = self.get_value_or_die(ori_node.input[3])
            assert tuple(axes) == (2, 3)
            ori_node.input.pop(3)

        mode = self.get_attribute(ori_node, "mode") or "nearest"
        ctm = self.get_attribute(ori_node, "coordinate_transformation_mode")
        nearest_mode = self.get_attribute(ori_node, "nearest_mode")
        cubic_coeff_a = self.get_attribute(ori_node, "cubic_coeff")
        pads_begin = self.get_attribute(ori_node, "pads_begin")
        pads_begin = list(map(int, pads_begin.split(",")))  # type: ignore
        pads_end = self.get_attribute(ori_node, "pads_end")
        pads_end = list(map(int, pads_end.split(",")))  # type: ignore

        if any(pads_begin) or any(pads_end):
            raise ValueError("Interpolate does not support converting with padding")

        return make_node(
            "Resize",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            mode=self._to_mode(mode),
            coordinate_transformation_mode=ctm or "half_pixel",
            nearest_mode=nearest_mode or "round_prefer_floor",
            cubic_coeff_a=cubic_coeff_a or -0.75,
        )
