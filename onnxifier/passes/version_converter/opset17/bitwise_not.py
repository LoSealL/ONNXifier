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

from typing import List

import numpy as np
from onnx.helper import make_node, np_dtype_to_tensor_dtype, tensor_dtype_to_np_dtype
from onnx.onnx_pb import NodeProto

from .... import OnnxGraph
from ...pattern import SingleNodePattern
from ...rewriter import Rewriter
from ...utils import cast_in, cast_out, make_constant
from . import OP_CONVERTER


@OP_CONVERTER.register("BitwiseNot")
class BitwiseNot(Rewriter):
    """Rewrite bitwise as arithmetic operators"""

    def __init__(self):
        super().__init__(SingleNodePattern("BitwiseNot"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        etype = graph.tensor_type(node.input[0])
        dtype = tensor_dtype_to_np_dtype(etype)
        utype = dtype.str.replace("i", "u")
        cst_node = make_constant(
            f"{node.name}/not", np.array(np.iinfo(utype).max, utype)
        )
        if dtype != utype:
            self += cast_in(node, 0, np_dtype_to_tensor_dtype(np.dtype(utype)))
        op_node = make_node(
            "Sub",
            inputs=[cst_node.output[0], node.input[0]],
            outputs=node.output,
            name=node.name,
        )
        if dtype != utype:
            self += cast_out(op_node, 0, np_dtype_to_tensor_dtype(dtype))
        self -= node
        self += [cst_node, op_node]
