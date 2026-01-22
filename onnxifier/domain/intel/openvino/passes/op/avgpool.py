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

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ...utils import text_to_integers
from .. import OnnxGraph
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="avg_pool")
class AvgPool(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/pooling/avg-pool-14.html

    https://onnx.ai/onnx/operators/onnx__AveragePool.html
    """

    def _trans_ceil_mode(self, ir_mode):
        if ir_mode == "ceil":
            return 1
        return 0

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        ceil_mode = self._trans_ceil_mode(self.get_attribute(ori_node, "rounding_type"))
        pads_begin = self.get_attribute(ori_node, "pads_begin")
        assert isinstance(pads_begin, str) or pads_begin is None
        pads_begin = text_to_integers(pads_begin)
        pads_end = self.get_attribute(ori_node, "pads_end")
        assert isinstance(pads_end, str) or pads_end is None
        pads_end = text_to_integers(pads_end)
        dilations = self.get_attribute(ori_node, "dilations")
        assert isinstance(dilations, str) or dilations is None
        dilations = text_to_integers(dilations)
        strides = self.get_attribute(ori_node, "strides")
        assert isinstance(strides, str) or strides is None
        strides = text_to_integers(strides)
        kernel = self.get_attribute(ori_node, "kernel")
        assert isinstance(kernel, str) or kernel is None
        kernel = text_to_integers(kernel)
        if auto_pad := self.get_attribute(ori_node, "auto_pad"):
            assert isinstance(auto_pad, str)
            if auto_pad == "explicit":
                auto_pad = "NOTSET"
                assert pads_begin is not None and pads_end is not None
                pads = list(itertools.chain(pads_begin, pads_end))
            else:
                pads = None
        else:
            auto_pad = "NOTSET"
            assert pads_begin is not None and pads_end is not None
            pads = list(itertools.chain(pads_begin, pads_end))
        return make_node(
            "AveragePool",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            auto_pad=auto_pad.upper(),
            ceil_mode=ceil_mode,
            dilations=dilations,
            pads=pads,
            strides=strides,
            kernel_shape=kernel,
        )
