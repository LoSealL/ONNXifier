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

import itertools
from typing import Any, Dict

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ...utils import text_to_integers
from .. import OnnxGraph, make_constant
from . import OP_CONVERT, BaseNodeConversion


def _trans_auto_pad(ir_str):
    match ir_str:
        case "explicit" | "notset" | None:
            return "NOTSET"
        case "auto":
            return "SAME_LOWER"
        case "same_lower":
            return "SAME_LOWER"
        case "same_upper":
            return "SAME_UPPER"
        case "valid":
            return "VALID"
    raise ValueError(f"Unknown auto_pad={ir_str}")


@OP_CONVERT.register(name="convolution")
class Convolution(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/convolution/convolution-1.html

    https://onnx.ai/onnx/operators/onnx__Conv.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        dilations = self.get_attribute(ori_node, "dilations")
        assert isinstance(dilations, str) or dilations is None
        dilations = text_to_integers(dilations)
        strides = self.get_attribute(ori_node, "strides")
        assert isinstance(strides, str) or strides is None
        strides = text_to_integers(strides)
        pads_begin = self.get_attribute(ori_node, "pads_begin")
        assert isinstance(pads_begin, str) or pads_begin is None
        pads_begin = text_to_integers(pads_begin)
        pads_end = self.get_attribute(ori_node, "pads_end")
        assert isinstance(pads_end, str) or pads_end is None
        pads_end = text_to_integers(pads_end)
        auto_pad = _trans_auto_pad(self.get_attribute(ori_node, "auto_pad"))
        attrs: Dict[str, Any] = dict(dilations=dilations, strides=strides)
        if auto_pad == "NOTSET":
            assert pads_begin is not None and pads_end is not None
            attrs["pads"] = list(itertools.chain(pads_begin, pads_end))
        else:
            attrs["auto_pad"] = auto_pad
        kernel_shape = graph.tensor_shape(ori_node.input[1])
        kernels = kernel_shape[2:]  # kernels could be 1D to 3D
        if all(isinstance(k, int) for k in kernels):
            attrs["kernel_shape"] = kernels
        return make_node(
            "Conv",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            **attrs,
        )


@OP_CONVERT.register(name="group_convolution")
class GroupConvolution(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/convolution/group-convolution-1.html

    https://onnx.ai/onnx/operators/onnx__Conv.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        dilations = self.get_attribute(ori_node, "dilations")
        assert isinstance(dilations, str) or dilations is None
        dilations = text_to_integers(dilations)
        strides = self.get_attribute(ori_node, "strides")
        assert isinstance(strides, str) or strides is None
        strides = text_to_integers(strides)
        pads_begin = self.get_attribute(ori_node, "pads_begin")
        assert isinstance(pads_begin, str) or pads_begin is None
        pads_begin = text_to_integers(pads_begin)
        pads_end = self.get_attribute(ori_node, "pads_end")
        assert isinstance(pads_end, str) or pads_end is None
        pads_end = text_to_integers(pads_end)
        auto_pad = _trans_auto_pad(self.get_attribute(ori_node, "auto_pad"))
        attrs: Dict[str, Any] = dict(dilations=dilations, strides=strides)
        if auto_pad == "NOTSET":
            assert pads_begin is not None and pads_end is not None
            attrs["pads"] = list(itertools.chain(pads_begin, pads_end))
        else:
            attrs["auto_pad"] = auto_pad
        # calculate groups
        weight_shape = graph.tensor_shape(ori_node.input[1])
        group = weight_shape[0]
        if isinstance(group, str) or group < 1:
            raise ValueError(f"Weight shape[0] is dynamic: {group}")
        attrs["group"] = group
        if group > 1:
            # should reshape weights from [G, C_out, C_in, K0, K1, ...]
            # to [G * C_out, C_in, K0, K1, ...].
            weights = self.get_value(ori_node.input[1])
            assert weights is not None
            weights = weights.reshape([-1, *weights.shape[2:]])
            cst_node = make_constant(f"{ori_node.name}/weights", weights)
            ori_node.input[1] = cst_node.output[0]
            weight_shape = weights.shape
            self += cst_node
        kernels = weight_shape[2:]
        if all(isinstance(k, int) for k in kernels):
            attrs["kernel_shape"] = kernels
        return make_node(
            "Conv",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            **attrs,
        )
