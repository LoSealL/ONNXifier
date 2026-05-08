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

from onnx import NodeProto, TensorProto

from .... import OnnxGraph, logger
from ....domain.trt.ops.recurrent_plugin import gated_delta_rule_schema
from ... import PASSES
from ...pattern import SingleNodePattern
from . import EnsureTensorRTDomain, elem_type_from_schema


@PASSES.register("trt_gated_delta_rule_replace")
class TrtGatedDeltaRuleReplaceRewriter(EnsureTensorRTDomain):
    """Replace functions with op_type starting with 'GatedDeltaRule' with
    TRT-specific GatedDeltaRule node.
    """

    def __init__(self):
        super().__init__(
            SingleNodePattern(gated_delta_rule_schema.name).with_domain("*")
        )

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto], *args, **kwargs):
        node = nodes[0]
        attrs = {}
        for name, attr in gated_delta_rule_schema.attributes.items():
            attrs[name] = self.get_attribute(node, attr.name, attr.default_value)
        q_shape, _ = graph.tensor_info(node.input[0])
        v_shape, v_type = graph.tensor_info(node.input[2])
        if "num_v_heads" not in attrs:
            if v_shape and len(v_shape) == 4:
                num_v_heads = v_shape[-2]
            else:
                logger.warning("Can not get attribute num_v_heads")
                return
        else:
            num_v_heads = attrs["num_v_heads"]
        if "v_dim" not in attrs:
            if v_shape and len(v_shape) == 4:
                head_v_dim = v_shape[-1]
            else:
                logger.warning("Can not get attribute v_dim")
                return
        else:
            head_v_dim = attrs["v_dim"]
        if "k_dim" not in attrs:
            k_shape, _ = graph.tensor_info(node.input[1])
            if k_shape and len(k_shape) == 4:
                head_k_dim = k_shape[-1]
            else:
                logger.warning("Can not get attribute k_dim")
                return
        else:
            head_k_dim = attrs["k_dim"]

        self.ensure_inputs_outputs(node, gated_delta_rule_schema)
        # ssm state
        node.input[5] = f"{node.name}/{gated_delta_rule_schema.inputs[5].name}"
        # context lengths
        node.input[6] = f"{gated_delta_rule_schema.inputs[6].name}"
        node.output[1] = f"{node.name}/{gated_delta_rule_schema.outputs[1].name}"
        batch = q_shape[0] if q_shape else 1
        ssm_state_shape = (batch, num_v_heads, head_k_dim, head_v_dim)
        if v_type == TensorProto.UNDEFINED:
            v_type = elem_type_from_schema(
                gated_delta_rule_schema, gated_delta_rule_schema.inputs[5].name
            )
        self.mark_graph_input(node.input[5], v_type, ssm_state_shape)
        self.mark_graph_input(
            node.input[6],
            elem_type_from_schema(gated_delta_rule_schema, node.input[6]),
            (batch,),
        )
        self.mark_graph_output(node.output[1], v_type, ssm_state_shape)
