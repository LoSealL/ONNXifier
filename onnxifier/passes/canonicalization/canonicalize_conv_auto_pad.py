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

from typing import List

from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register(name="canonicalize_conv_autopad")
class CanonicalizeConvAutoPadRewriter(Rewriter):
    """After onnxsim it may combine pad into Conv, but still keep auto_pad attribute.
    This will cause an error in onnxruntime. So we need to remove the auto_pad
    attribute if pads is also set.
    """

    def __init__(self):
        super().__init__(
            SingleNodePattern("Conv").with_attr("auto_pad").with_attr("pads")
        )

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        auto_pad = self.get_attribute(node, "auto_pad")
        if auto_pad != "NOTSET":
            self.set_attribute(node, "auto_pad", "NOTSET")
