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

import numpy as np
from onnx import NodeProto, TensorProto
from onnx.helper import tensor_dtype_to_np_dtype

from .... import OnnxGraph, logger
from ....domain.trt.ops.mamba_plugin import causal_conv1d_schema
from ... import PASSES
from ...pattern import SingleNodePattern
from ...utils import make_constant
from . import EnsureTensorRTDomain, elem_type_from_schema


@PASSES.register("trt_causal_conv1d_replace")
class TrtCausalConv1dReplaceRewriter(EnsureTensorRTDomain):
    """Replace functions with op_type starting with 'CausalConv1d' with
    TRT-specific GatedDeltaNetCausalConv1d node.
    """

    def __init__(self):
        super().__init__(SingleNodePattern(causal_conv1d_schema.name).with_domain("*"))

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto], *args, **kwargs):
        node = nodes[0]
        # change conv_state and context_lengths as graph input
        self.ensure_inputs_outputs(node, causal_conv1d_schema)
        # conv state
        node.input[3] = f"{node.name}/{causal_conv1d_schema.inputs[3].name}"
        # context lengths
        node.input[4] = f"{causal_conv1d_schema.inputs[4].name}"
        node.output[1] = f"{node.name}/{causal_conv1d_schema.outputs[1].name}"
        x_shape, x_dtype = graph.tensor_info(node.input[0])
        if x_dtype == TensorProto.UNDEFINED:
            x_dtype = elem_type_from_schema(
                causal_conv1d_schema, causal_conv1d_schema.inputs[0].name
            )
        batch = x_shape[0] if x_shape else 1
        weight_shape = graph.tensor_shape(node.input[1])
        conv_state_shape = (batch, weight_shape[0], weight_shape[-1])
        self.mark_graph_input(
            node.input[4],
            elem_type_from_schema(causal_conv1d_schema, node.input[4]),
            (batch,),
        )
        self.mark_graph_input(node.input[3], x_dtype, conv_state_shape)
        self.mark_graph_output(node.output[1], x_dtype, conv_state_shape)

    def _check_bias(self, graph: OnnxGraph, node: NodeProto):
        if len(graph.onnx_predecessors(node)) >= 3:
            return  # contain bias
        # try get weight shape
        weight_shape, w_type = graph.tensor_info(node.input[1])
        if w_type > 0:
            dtype = tensor_dtype_to_np_dtype(w_type)
        else:
            dtype = np.float16
        if not weight_shape:
            weight_value = self.get_value(node.input[1])
            if weight_value is None:
                logger.warning("can't get weight shape, bias is required but missing")
                return False
            weight_shape = weight_value.shape
            dtype = weight_value.dtype
        channels = weight_shape[0]
        if not isinstance(channels, int):
            logger.warning("can't get weight channels, bias is required but missing")
            return False
        bias = make_constant(f"{node.name}/bias", np.zeros([channels], dtype))
        node.input[2] = bias.output[0]
        self += bias
