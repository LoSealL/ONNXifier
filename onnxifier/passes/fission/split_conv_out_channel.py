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
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph, logger
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register("split_conv_out_channel", deps=["infer_shape"])
class SplitConvOutChannelRewriter(Rewriter):
    r"""Split the output channel of Conv into multiple Conv nodes.

    Before:

        Conv

    After:

        Conv---
        Conv---|
        Conv---|--Concat
        ...    |
        Conv---

    Args:
        min_out_channels: Ignore if the output channel is less than this value.
        target: target number of output channels for each split Conv node.
    """

    def __init__(self, min_out_channels: int = 256, target: int = 128):
        super().__init__(SingleNodePattern("Conv"))
        self.min_out_channels = int(min_out_channels)
        self.target = target

    def _conv_attrs(self, conv):
        keys = ("auto_pad", "dilations", "group", "kernel_shape", "pads", "strides")
        attrs = {k: self.get_attribute(conv, k) for k in keys}
        return {k: v for k, v in attrs.items() if v is not None}

    def _split_conv(self, node, weight_value, bias_value):
        group = self.get_attribute(node, "group") or 1
        assert isinstance(group, int)
        if group > 1:
            assert weight_value.shape[1] == 1, "not a depthwise convolution"
        out_channels = weight_value.shape[0]
        channel_splits = [self.target] * (out_channels // self.target)
        if sum(channel_splits) != out_channels:
            channel_splits.append(out_channels - sum(channel_splits))
        logger.debug(
            f"split conv {node.name} with out_channels "
            f"{out_channels} to {channel_splits}"
        )
        channel_indices = np.cumsum(channel_splits)
        weight_splits = np.split(weight_value, channel_indices[:-1], axis=0)
        if bias_value is not None:
            bias_splits = np.split(bias_value, channel_indices[:-1], axis=0)
        else:
            bias_splits = []
        convs = []
        channel_indices = [0] + channel_indices.tolist()
        for i, _ in enumerate(channel_splits):
            new_weight = make_constant(f"{node.name}/weight{i}", weight_splits[i])
            self += new_weight
            if bias_value is not None:
                new_bias = make_constant(f"{node.name}/bias{i}", bias_splits[i])
                self += new_bias
                bias = [new_bias.output[0]]
            else:
                bias = []
            new_conv = make_node(
                "Conv",
                inputs=[node.input[0], new_weight.output[0]] + bias,
                outputs=[f"{node.output[0]}_{i}"],
                name=f"{node.name}_{i}",
                **self._conv_attrs(node),  # type: ignore
            )
            if group > 1:
                self.set_attribute(new_conv, "group", channel_splits[i])
                start = make_constant(
                    f"{node.name}/start{i}", np.array([channel_indices[i]])
                )
                end = make_constant(
                    f"{node.name}/end{i}", np.array([channel_indices[i + 1]])
                )
                axes = make_constant(f"{node.name}/axes{i}", np.array([1]))
                self += [start, end, axes]
                self += make_node(
                    "Slice",
                    inputs=[
                        node.input[0],
                        start.output[0],
                        end.output[0],
                        axes.output[0],
                    ],
                    outputs=[f"{node.output[0]}/Slice{i}"],
                )
                new_conv.input[0] = f"{node.output[0]}/Slice{i}"
            convs.append(new_conv)
        concat = make_node(
            "Concat",
            inputs=[c.output[0] for c in convs],
            outputs=[node.output[0]],
            name=f"{node.name}/Concat",
            axis=1,
        )
        self += convs
        self += concat
        self -= node

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        # overwrite default parameters
        self.min_out_channels = kwargs.get("min_out_channels", self.min_out_channels)
        self.target = kwargs.get("target", self.target)

        node = nodes[0]
        weight_value = self.get_value(node.input[1])
        bias_value = None
        if len(node.input) == 3:
            bias_value = self.get_value(node.input[2])
        if weight_value is None:
            # dynamic weight
            logger.debug(f"skip conv {node.name} with dynamic weight")
            return
        if weight_value.shape[0] <= self.min_out_channels:
            logger.debug(
                f"skip conv {node.name} with out_channels {weight_value.shape[0]}"
            )
            return
        if group := self.get_attribute(node, "group"):
            assert isinstance(group, int)
            if group > 1 and group != weight_value.shape[0]:
                return
        self._split_conv(node, weight_value, bias_value)
