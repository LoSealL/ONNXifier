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
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from .. import OnnxGraph, make_constant
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="pad")
class Pad(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/pad-12.html

    https://onnx.ai/onnx/operators/onnx__Pad.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        mode = self.get_attribute(ori_node, "pad_mode") or "constant"
        if mode in ("symmetric",):
            raise ValueError(f"Pad {ori_node.name} has unsupported pad mode: {mode}")

        begin = self.get_value_or_die(ori_node.input[1])
        end = self.get_value_or_die(ori_node.input[2])
        if len(ori_node.input) > 3:
            const_value = self.get_value(ori_node.input[3])
            if const_value is not None:
                ori_node.input[2] = ori_node.input[3]
            ori_node.input.pop(3)
        # pads must be int64
        pads = make_constant(
            f"{ori_node.name}/pads", np.concatenate([begin, end]).astype(np.int64)
        )
        ori_node.input[1] = pads.output[0]
        ori_node.input.pop(2)
        self += pads
        return make_node(
            "Pad",
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
            mode=mode,
        )
