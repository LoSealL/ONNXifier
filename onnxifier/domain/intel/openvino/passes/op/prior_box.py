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

from onnx import numpy_helper
from onnx.helper import make_node, tensor_dtype_to_np_dtype
from onnx.onnx_pb import NodeProto
from openvino import Model, compile_model
from openvino.opset8 import constant, prior_box

from ...utils import text_to_boolean
from .. import OnnxGraph
from . import OP_CONVERT, BaseNodeConversion


def _to_floats(text):
    if not text:
        return []
    return list(map(float, text.split(",")))


@OP_CONVERT.register(name="prior_box")
class PriorBox(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/detection/prior-box-8.html

    Fold to constant
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        layer_shape = self.get_value(ori_node.input[0])
        image_shape = self.get_value(ori_node.input[1])
        if layer_shape is None or image_shape is None:
            return ori_node
        out_shape, out_dtype = graph.tensor_info(ori_node.output[0])
        layer_shape = constant(layer_shape, name="layer_shape")
        image_shape = constant(image_shape, name="image_shape")

        pb_attrs = dict(
            min_size=_to_floats(self.get_attribute(ori_node, "min_size")),
            max_size=_to_floats(self.get_attribute(ori_node, "max_size")),
            aspect_ratio=_to_floats(self.get_attribute(ori_node, "aspect_ratio")),
            density=_to_floats(self.get_attribute(ori_node, "density")),
            fixed_ratio=_to_floats(self.get_attribute(ori_node, "fixed_ratio")),
            fixed_size=_to_floats(self.get_attribute(ori_node, "fixed_size")),
            clip=text_to_boolean(self.get_attribute(ori_node, "clip")),  # type: ignore
            flip=text_to_boolean(self.get_attribute(ori_node, "flip")),  # type: ignore
            step=float(self.get_attribute(ori_node, "step")),  # type: ignore
            offset=float(self.get_attribute(ori_node, "offset") or 0),  # type: ignore
            variance=_to_floats(self.get_attribute(ori_node, "variance")),
            scale_all_sizes=text_to_boolean(
                self.get_attribute(ori_node, "scale_all_sizes")  # type: ignore
            ),
        )
        if order := self.get_attribute(ori_node, "min_max_aspect_ratios_order"):
            assert isinstance(order, str)
            pb_attrs["min_max_aspect_ratios_order"] = text_to_boolean(order)
        prior_node = prior_box(layer_shape, image_shape, pb_attrs)
        model = Model([prior_node], [])  # type: ignore
        compiled_model = compile_model(model, "CPU")
        value = compiled_model()[-1]
        assert out_shape is not None and value.shape == tuple(out_shape)
        assert value.dtype == tensor_dtype_to_np_dtype(out_dtype)

        return make_node(
            "Constant",
            [],
            outputs=ori_node.output,
            name=ori_node.name,
            value=numpy_helper.from_array(value),
        )
