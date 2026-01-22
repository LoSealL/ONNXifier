"""
Copyright (C) 2026 The ONNXIFIER Authors.

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

# pylint: disable=arguments-differ

from typing import List

import numpy as np
from onnx.helper import make_node, tensor_dtype_to_np_dtype
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(name="space2depth_to_conv")
class SpaceToDepthToConvRewriter(Rewriter):
    """Convert SpaceToDepth to Conv."""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("SpaceToDepth"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], to_depthwise=False):
        s2d = nodes[0]
        blocksize = self.get_attribute(s2d, "blocksize")
        assert isinstance(blocksize, int)
        _, ic, _, _ = graph.tensor_shape(s2d.input[0])
        _, oc, _, _ = graph.tensor_shape(s2d.output[0])
        assert isinstance(ic, int) and isinstance(oc, int)
        dtype = graph.tensor_type(s2d.input[0])
        assert ic * blocksize**2 == oc, "invalid space2depth parameters"

        assert not to_depthwise, "not implemented yet"
        if to_depthwise:
            kernel = np.tile(
                np.eye(blocksize**2)
                .reshape([-1, blocksize, blocksize])
                .astype(tensor_dtype_to_np_dtype(dtype)),
                [ic, 1, 1],
            )[:, None]
            # TODO: output channels need to be reordered
        else:
            kernel = (
                np.tile(
                    np.eye(blocksize**2)
                    .reshape([-1, blocksize, blocksize])
                    .astype(tensor_dtype_to_np_dtype(dtype)),
                    [ic, 1, 1, 1],
                )
                .transpose([1, 0, 2, 3])
                .reshape([oc, 1, blocksize, blocksize])
            )
            # expand to [oc, ic, blocksize, blocksize]
            kernel = np.concatenate(
                [kernel, np.tile(np.zeros_like(kernel), [1, ic - 1, 1, 1])], axis=1
            )
            for i in range(1, ic):
                kernel[i::ic][:, i] = kernel[i::ic][:, 0]
                kernel[i::ic][:, 0] = 0  # swap
        conv_weight = make_constant(name=f"{s2d.name}/weight", value=kernel)
        conv = make_node(
            op_type="Conv",
            inputs=list(s2d.input) + [conv_weight.output[0]],
            outputs=s2d.output,
            name=f"{s2d.name}/conv",
            kernel_shape=[blocksize, blocksize],
            strides=[blocksize, blocksize],
            group=ic if to_depthwise else 1,
        )
        self -= s2d
        self += [conv, conv_weight]
