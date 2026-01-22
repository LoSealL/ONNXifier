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

from .... import OnnxGraph
from ...pattern import SingleNodePattern
from ...rewriter import Rewriter
from ...utils import make_constant
from . import OP_CONVERTER


@OP_CONVERTER.register("LayerNormalization")
class LayerNormalization(Rewriter):
    """Decompose LayerNormalization to ReduceMean and Subtract.

    LN(X) = (X - E[X]) / sqrt(Var[X] + epsilon) * Scale + Bias
    """

    def __init__(self):
        super().__init__(SingleNodePattern("LayerNormalization"))

    def _var_x(self, node: NodeProto):
        # VAR(X) = ReduceMean((X - ReduceMean(x, axis))^2, axis)
        axis = self.get_attribute(node, "axis", default=-1)
        epsilon = self.get_attribute(node, "epsilon", default=1e-5)
        assert isinstance(axis, int) and isinstance(epsilon, float)
        rank = len(self.graph.tensor_shape(node.input[0]))
        if axis < 0:
            axis += rank
        axes = list(range(axis, rank))
        mean_x = make_node(
            "ReduceMean",
            [node.input[0]],
            [f"{node.name}/ReduceMean_X_output0"],
            name=f"{node.name}/ReduceMean_X",
            axes=axes,
            keepdims=1,
        )
        if len(node.output) >= 2:
            mean_x.output[0] = node.output[1]
        sub_x_mean = make_node(
            "Sub",
            [node.input[0], mean_x.output[0]],
            [f"{node.name}/Sub_X_Mean_output0"],
            name=f"{node.name}/Sub_X_Mean",
        )
        square_d = make_node(
            "Mul",
            [sub_x_mean.output[0], sub_x_mean.output[0]],
            [f"{node.name}/Square_D_output0"],
            name=f"{node.name}/Square_D",
        )
        var_x = make_node(
            "ReduceMean",
            [square_d.output[0]],
            [f"{node.name}/ReduceMean_Var_X_output0"],
            name=f"{node.name}/ReduceMean_Var_X",
            axes=axes,
            keepdims=1,
        )
        epsilon_cst = make_constant(
            f"{node.name}/epsilon", np.array(epsilon, dtype=np.float32)
        )
        var_eps = make_node(
            "Add",
            [var_x.output[0], epsilon_cst.output[0]],
            [f"{node.name}/Add_Var_Eps_output0"],
            name=f"{node.name}/Add_Var_Eps",
        )
        self += [mean_x, sub_x_mean, square_d, var_x, epsilon_cst, var_eps]
        return sub_x_mean, var_eps

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        has_bias = len(node.input) == 3
        delta, var_x_eps = self._var_x(node)
        std_x = make_node(
            "Sqrt",
            [var_x_eps.output[0]],
            [f"{node.name}/Std_Var_output0"],
            name=f"{node.name}/Std_Var",
        )
        inv_std_x = make_node(
            "Reciprocal",
            [std_x.output[0]],
            [f"{node.name}/Inv_Std_output0"],
            name=f"{node.name}/Inv_Std",
        )
        if len(node.output) == 3:
            inv_std_x.output[0] = node.output[2]
        ln_x = make_node(
            "Mul",
            [delta.output[0], inv_std_x.output[0]],
            [f"{node.name}/Div_Std_output0"],
            name=f"{node.name}/Div_Std",
        )
        scale_x = make_node(
            "Mul",
            [ln_x.output[0], node.input[1]],
            [f"{node.name}/Scale_output0"],
            name=f"{node.name}/Scale",
        )
        self += [std_x, inv_std_x, ln_x, scale_x]
        if has_bias:
            bias_x = make_node(
                "Add",
                [scale_x.output[0], node.input[2]],
                [f"{node.name}/Bias_output0"],
                name=f"{node.name}/Bias",
            )
            self += bias_x
            bias_x.output[0] = node.output[0]
        else:
            scale_x.output[0] = node.output[0]
        self -= node
