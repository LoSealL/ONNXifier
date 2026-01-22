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

from ...utils import text_to_boolean, text_to_integers
from .. import OnnxGraph, make_constant
from . import OP_CONVERT, BaseNodeConversion


@OP_CONVERT.register(name="mvn")
class MVN(BaseNodeConversion):
    """https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/normalization/mvn-6.html

    https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
    """

    def replace(self, graph: OnnxGraph, ori_node: NodeProto) -> NodeProto:
        normalize_variance = self.get_attribute(ori_node, "normalize_variance")
        assert isinstance(normalize_variance, str)
        normalize_variance = text_to_boolean(normalize_variance)
        eps = float(self.get_attribute(ori_node, "eps"))  # type: ignore
        eps_mode = self.get_attribute(ori_node, "eps_mode") or "inside_sqrt"
        assert isinstance(eps_mode, str)
        eps_mode = eps_mode.lower()
        assert eps_mode in ("inside_sqrt", "outside_sqrt")

        if len(ori_node.input) == 1:
            # backward compatibility for mvn-1
            across_channels = self.get_attribute(ori_node, "across_channels")
            assert isinstance(across_channels, str)
            across_channels = text_to_boolean(across_channels)
            reduction_axes = self.get_attribute(ori_node, "reduction_axes")
            if across_channels:
                input_rank = len(graph.tensor_shape(ori_node.input[0]))
                axes = np.arange(input_rank, dtype="int64")
            else:
                assert isinstance(reduction_axes, str)
                axes = np.array(text_to_integers(reduction_axes), "int64")
        else:
            axes = self.get_value_or_die(ori_node.input[1]).astype("int64")
        axes_norm = np.arange(axes.min(), axes.max() + 1).astype("int64")
        if len(axes) <= 1:
            rank = len(graph.tensor_shape(ori_node.input[0]))
            axes_norm = np.arange(axes.min(), rank)
        if not np.all(axes == axes_norm) and len(axes) > 1:
            raise ValueError(f"Unsupported axes: {axes}")
        # TODO: axes_norm is no needed, can be replaced by axes after 2024.3
        axes_node = make_constant(f"{ori_node.name}/axes", axes_norm)
        r_mean = make_node(
            "ReduceMean",
            [ori_node.input[0], axes_node.output[0]],
            [f"{ori_node.name}/ReduceMean_output0"],
            name=f"{ori_node.name}/ReduceMean",
            keepdims=1,
        )

        sub = make_node(
            "Sub",
            [ori_node.input[0], r_mean.output[0]],
            [f"{ori_node.name}/Sub_output0"],
            name=f"{ori_node.name}/Sub",
        )

        if not normalize_variance:
            self += [axes_node, r_mean]
            sub.name = ori_node.name
            sub.output[0] = ori_node.output[0]
            return sub
        if eps_mode == "inside_sqrt" and np.all(axes == axes_norm):
            # NOTE: OpenVINO handles LayerNormalization differently with Onnxruntime.
            # In Onnxruntime, `scale` must match the size of normalized layer size.
            # For example, a tensor [1, 32, 8, 8] with `axis=2`, its layer size is 64,
            # so `scale` can be a shape of [64] or [8, 8] or even [1, 64].
            # While in OpenVINO, `scale` must be broadcastable to the layer size. so
            # `scale` must be a shape of [8, 8] in this case.
            axis = int(axes_norm.flatten()[0])
            scale_s = graph.static_tensor_shape(ori_node.output[0])[axis:]
            scale = make_constant(
                f"{ori_node.name}/scale", np.ones(scale_s, np.float32)  # type: ignore
            )
            self += scale
            return make_node(
                "LayerNormalization",
                inputs=[ori_node.input[0], scale.output[0]],
                outputs=[ori_node.output[0]],
                name=ori_node.name,
                axis=axis,
                epsilon=eps,
            )
        mul = make_node(
            "Mul",
            [sub.output[0], sub.output[0]],
            [f"{ori_node.name}/Mul_output0"],
            name=f"{ori_node.name}/Mul",
        )
        r_mean2 = make_node(
            "ReduceMean",
            [mul.output[0], axes_node.output[0]],
            [f"{ori_node.name}/ReduceMean2_output0"],
            name=f"{ori_node.name}/ReduceMean2",
            keepdims=1,
        )
        sqrt = make_node(
            "Sqrt",
            [r_mean2.output[0]],
            [f"{ori_node.name}/Sqrt_output0"],
            name=f"{ori_node.name}/Sqrt",
        )
        eps_node = make_constant(f"{ori_node.name}/eps", np.array(eps, np.float32))
        add_eps = make_node(
            "Add",
            [sqrt.output[0], eps_node.output[0]],
            [f"{ori_node.name}/Add_output0"],
            name=f"{ori_node.name}/Add",
        )
        self += [axes_node, r_mean, sub, mul, r_mean2, eps_node, add_eps, sqrt]
        return make_node(
            "Div",
            inputs=[sub.output[0], add_eps.output[0]],
            outputs=[ori_node.output[0]],
            name=ori_node.name,
        )
