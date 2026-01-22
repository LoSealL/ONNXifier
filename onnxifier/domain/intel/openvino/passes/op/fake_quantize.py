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

import numpy as np
from onnx.helper import make_node, tensor_dtype_to_np_dtype
from onnx.onnx_pb import NodeProto

from .. import OnnxGraph, make_constant
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="fake_quantize")
class FakeQuantize(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/quantization/fake-quantize-1.html

    Split fake quantize to QuantizeLinear and DequantizeLinear.
    https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html
    https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html

    Note:

        FakeQuantize only applies to activations, not weights.
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        levels = int(self.get_attribute(ori_node, "levels") or 256)  # type: ignore
        nbits = int(np.log2(levels))
        assert nbits in (8, 16, 32)
        input_low = self.get_value_or_die(ori_node.input[1]).squeeze()
        input_high = self.get_value_or_die(ori_node.input[2]).squeeze()
        output_low = self.get_value_or_die(ori_node.input[3]).squeeze()
        output_high = self.get_value_or_die(ori_node.input[4]).squeeze()

        try:
            scale_type = graph.tensor_type(ori_node.output[0])
            scale_prec = tensor_dtype_to_np_dtype(scale_type)
        except ValueError:
            scale_prec = np.float32
        zero_prec = np.iinfo(f"uint{nbits}").dtype
        scales = ((input_high - input_low) / (levels - 1)).astype(scale_prec)
        zero_points = np.rint((levels - 1) * input_low / (input_low - input_high))
        zero_points = zero_points.astype(zero_prec)
        scale_node = make_constant(f"{ori_node.name}/QuantizeLinear_y_scale", scales)
        zp_node = make_constant(
            f"{ori_node.name}/QuantizeLinear_y_zero_point", zero_points
        )
        # add QuantizeLinear
        quantizelinear = make_node(
            "QuantizeLinear",
            inputs=[ori_node.input[0], scale_node.output[0], zp_node.output[0]],
            outputs=[f"{ori_node.name}/QuantizeLinear_output0"],
            name=f"{ori_node.name}/QuantizeLinear",
        )
        self += [scale_node, zp_node, quantizelinear]
        # replace self with DequantizeLinear
        scales = ((output_high - output_low) / (levels - 1)).astype(scale_prec)
        zero_points = np.rint((levels - 1) * output_low / (output_low - output_high))
        zero_points = zero_points.astype(zero_prec)
        scale_node = make_constant(f"{ori_node.name}/DequantizeLinear_y_scale", scales)
        zp_node = make_constant(
            f"{ori_node.name}/DequantizeLinear_y_zero_point", zero_points
        )
        self += [scale_node, zp_node]
        return make_node(
            "DequantizeLinear",
            inputs=[quantizelinear.output[0], scale_node.output[0], zp_node.output[0]],
            outputs=ori_node.output,
            name=ori_node.name,
        )
