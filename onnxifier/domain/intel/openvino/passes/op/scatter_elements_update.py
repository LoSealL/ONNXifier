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


@OP_CONVERT.register(name="scatter_elements_update")
class ScatterElementsUpdate(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/scatter-elements-update-12.html

    https://onnx.ai/onnx/operators/onnx__ScatterElements.html
    """

    _reduction_map = {
        "none": "none",
        "sum": "add",
        "prod": "mul",
        "min": "min",
        "max": "max",
    }

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        axis = self.get_value_or_die(ori_node.input[3])
        reduction = self.get_attribute(ori_node, "reduction")
        assert isinstance(reduction, str)
        if reduction == "mean":
            raise ValueError("Unsupported reduction: mean")
        reduction = self._reduction_map[reduction]
        return make_node(
            "ScatterElements",
            inputs=ori_node.input[:3],
            outputs=ori_node.output,
            name=ori_node.name,
            axis=int(axis),
            reduction=reduction,
        )


@OP_CONVERT.register(name="scatter_update")
class ScatterUpdate(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/movement/scatter-update-3.html

    https://onnx.ai/onnx/operators/onnx__ScatterElements.html

    Note:

        We should convert to ScatterElements since Scatter is deprecated since opset 11.
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        indices_shape = graph.static_tensor_shape(ori_node.input[1])
        updates_shape = graph.static_tensor_shape(ori_node.input[2])
        axis = self.get_value_or_die(ori_node.input[3]).squeeze()

        if axis < 0:
            axis += len(graph.tensor_shape(ori_node.input[0]))
        r = len(graph.tensor_shape(ori_node.input[1]))
        if r > 1:  # insert reshape
            # fmt: off
            s = np.prod(updates_shape[axis:axis + r])
            updates_shape = updates_shape[:axis] + [s] + updates_shape[axis + r:]
            # fmt: on
            updates_shape_cst = make_constant(
                f"{ori_node.name}/updates_shape", np.array(updates_shape, np.int64)
            )
            reshape = make_node(
                "Reshape",
                inputs=[ori_node.input[2], updates_shape_cst.output[0]],
                outputs=[f"{ori_node.name}/updates/Reshape_output0"],
                name=f"{ori_node.name}/updates/Reshape",
            )
            ori_node.input[2] = reshape.output[0]
            self += [updates_shape_cst, reshape]
        # tile indices
        new_indices_shape = [
            1 for _ in range(len(graph.tensor_shape(ori_node.input[0])))
        ]
        new_indices_shape[axis] = int(np.prod(indices_shape))
        indices_shape_cst = make_constant(
            f"{ori_node.name}/indices_shape", np.array(new_indices_shape, np.int64)
        )
        indices_reshape = make_node(
            "Reshape",
            inputs=[ori_node.input[1], indices_shape_cst.output[0]],
            outputs=[f"{ori_node.name}/indices/Reshape_output0"],
            name=f"{ori_node.name}/indices/Reshape",
        )
        tile_shape = (np.array(updates_shape) // new_indices_shape).astype(np.int64)
        tile_shape_cst = make_constant(f"{ori_node.name}/tile_shape", tile_shape)
        indices_tile = make_node(
            "Tile",
            inputs=[indices_reshape.output[0], tile_shape_cst.output[0]],
            outputs=[f"{ori_node.name}/indices/Tile_output0"],
            name=f"{ori_node.name}/indices/Tile",
        )
        ori_node.input[1] = indices_tile.output[0]
        self += [indices_shape_cst, tile_shape_cst, indices_reshape, indices_tile]
        return make_node(
            "ScatterElements",
            inputs=ori_node.input[:3],
            outputs=ori_node.output,
            name=ori_node.name,
            axis=int(axis),
        )
