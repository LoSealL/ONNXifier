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

from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from .. import OnnxGraph, SingleNodePattern
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="z__trivial")  # make sure to be called in the last
class Trivial(BaseNodeConversion):
    """Trivial conversion for OpenVINO nodes."""

    def __init__(self):
        super().__init__(SingleNodePattern().with_attr("version"))

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        # replace ai.intel.openvino domain to ai.onnx (empty)
        if ori_node.domain != "ai.intel.openvino":
            raise RuntimeError(
                f"node {ori_node.name} got an unexpected domain: {ori_node.domain}"
            )
        return make_node(
            ori_node.op_type,
            inputs=ori_node.input,
            outputs=ori_node.output,
            name=ori_node.name,
        )
