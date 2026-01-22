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

import numpy as np
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from .... import OnnxGraph
from ...pattern import SingleNodePattern
from ...rewriter import Rewriter
from ...utils import make_constant
from . import OP_CONVERTER


@OP_CONVERTER.register("CenterCropPad")
class CenterCropPad(Rewriter):
    """CenterCropPad to pad and slice"""

    def __init__(self):
        super().__init__(SingleNodePattern("CenterCropPad"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        axes = self.get_attribute(node, "axes")
        shape = self.get_value(node.input[1])
        if shape is None:
            raise RuntimeError(f"shape of {node.name} is not constant")
        ori_shape = graph.static_tensor_shape(node.input[0])
        if axes is None:
            axes = list(range(len(shape)))
        axes = sorted(axes)  # type: ignore

        if self._need_pad(shape, ori_shape, axes):
            shape_iter = iter(shape)
            shape_expand = list(ori_shape)
            for i in axes:
                shape_expand[i] = next(shape_iter)
            pad_list = [max(0, i - j) for i, j in zip(shape_expand, ori_shape)]
            paddings = np.array([[i // 2, i - i // 2] for i in pad_list], np.int64)
            paddings = paddings.transpose().flatten()
            pad_cst = make_constant(f"{node.name}/Pad/pads", paddings)
            pad = make_node(
                "Pad",
                inputs=[node.input[0], pad_cst.output[0]],
                outputs=[f"{node.name}/Pad_output0"],
                name=f"{node.name}/Pad",
            )
            self += [pad_cst, pad]
            node.input[0] = pad.output[0]
            ori_shape = [max(i, j) for i, j in zip(shape_expand, ori_shape)]
        # get slice begin and end
        begin, end = self._slice_begin_and_end(shape, ori_shape, axes)
        beg_node = make_constant(f"{node.name}/Slice/begin", np.array(begin, np.int64))
        end_node = make_constant(f"{node.name}/Slice/end", np.array(end, np.int64))
        axes_node = make_constant(f"{node.name}/Slice/axes", np.array(axes, np.int64))
        slice_node = make_node(
            "Slice",
            inputs=[
                node.input[0],
                beg_node.output[0],
                end_node.output[0],
                axes_node.output[0],
            ],
            outputs=node.output,
            name=node.name,
        )
        self += [beg_node, end_node, axes_node, slice_node]
        self -= node

    def _need_pad(self, shape, ori_shape, axes):
        shape_iter = iter(shape)
        return any(next(shape_iter) > ori_shape[i] for i in axes)

    def _slice_begin_and_end(self, shape, ori_shape, axes):
        shape_iter = iter(shape)
        slices = [ori_shape[i] - next(shape_iter) for i in axes]
        begin = [s // 2 for s in slices]
        end = [ori_shape[i] - s // 2 for i, s in zip(axes, slices)]
        return begin, end
