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

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from .. import OnnxGraph, make_constant
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="broadcast")
class Broadcast(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/broadcast-3.html

    Use Reshape and Tile to implement broadcast.
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        mode = self.get_attribute(ori_node, "mode")
        if mode not in ("numpy", "explicit", "bidirectional", ""):
            raise NotImplementedError(f"Unsupported mode: {mode}")

        input_shape = graph.tensor_shape(ori_node.input[0])
        input_rank = len(input_shape)
        target_shape = self.get_value(ori_node.input[1])
        if target_shape is None:
            # to expand
            return make_node(
                "Expand",
                inputs=ori_node.input[:2],
                outputs=ori_node.output,
                name=ori_node.name,
            )
        if mode == "bidirectional":
            # recalculate target shape
            for k, (i, j) in enumerate(
                zip(reversed(input_shape), reversed(target_shape))
            ):
                if i != j and i != 1:
                    assert j == 1
                    target_shape[len(target_shape) - k - 1] = i
        elif mode == "explicit":
            axes_mapping = self.get_value_or_die(ori_node.input[2])
            ascending_axes = np.sort(axes_mapping)
            if not np.all(axes_mapping == ascending_axes):
                raise ValueError(
                    f"Expect axes_mapping to be ascending, but got {axes_mapping}"
                )
            shape = np.ones_like(target_shape, dtype=np.int64)
            shape[axes_mapping] = input_shape
            shape_node = make_constant(f"{ori_node.name}/shape", shape)
            reshape = make_node(
                "Reshape",
                inputs=[ori_node.input[0], shape_node.output[0]],
                outputs=[ori_node.input[0] + "/Reshape"],
                name=f"{ori_node.name}/Reshape",
            )
            ori_node.input[0] = reshape.output[0]
            self += [shape_node, reshape]
            input_shape = shape
            input_rank = len(input_shape)
        if input_rank != len(target_shape):
            # insert Unsqueeze to add missing dimensions
            shape = graph.static_tensor_shape(ori_node.input[0])
            shape = [1] * (len(target_shape) - input_rank) + shape
            shape_node = make_constant(
                f"{ori_node.name}/shape", np.array(shape, dtype=np.int64)
            )
            reshape = make_node(
                "Reshape",
                inputs=[ori_node.input[0], shape_node.output[0]],
                outputs=[ori_node.input[0] + "/Reshape"],
                name=f"{ori_node.name}/Reshape",
            )
            ori_node.input[0] = reshape.output[0]
            self += [shape_node, reshape]
            input_shape = shape
        # translate broadcast shape to Tile's repeats
        if np.any(target_shape % input_shape):
            raise ValueError(
                f"Broadcast shape ({target_shape}) is not compatible with "
                f"input shape ({input_shape})."
            )
        repeats = target_shape // input_shape
        repeats_node = make_constant(
            f"{ori_node.name}/repeats", repeats.astype(np.int64)
        )
        ori_node.input[1] = repeats_node.output[0]
        self += repeats_node

        return make_node(
            "Tile",
            inputs=ori_node.input[:2],
            outputs=ori_node.output,
            name=ori_node.name,
        )
