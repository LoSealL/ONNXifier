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

# pylint: disable=arguments-differ
from typing import Dict, List, Optional

from onnx import NodeProto, shape_inference
from onnx.tools.update_model_dims import update_inputs_outputs_dims

from ...graph import OnnxGraph
from ...logger import debug, warning
from .. import PASSES, Registry, Rewriter, get_pass_manager
from ..pattern import SingleNodePattern

INFERSHAPE_PATCH = Registry("INFERSHAPE_PATCH", parent=PASSES)


@INFERSHAPE_PATCH.register()
class InferSplitToSequence(Rewriter):
    """[BUG] inference shape after SplitToSequence is incorrect.

    Related github issue: [#6656](https://github.com/onnx/onnx/issues/6656)
    """

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("SplitToSequence"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        input_shape = graph.static_tensor_shape(node.input[0])
        axis = self.get_attribute(node, "axis", 0)
        keepdims = self.get_attribute(node, "keepdims", 1)
        assert isinstance(axis, int) and isinstance(keepdims, int)
        axis = int(axis)
        keepdims = bool(keepdims)
        if len(node.input) > 1:
            split_arr = self.get_value_or_die(node.input[1])
            if split_arr.ndim == 0:
                split = [int(split_arr)] * (input_shape[axis] // split_arr)
            else:
                split = [int(i) for i in split_arr]
            assert sum(split) == input_shape[axis]
        else:
            split = [1] * input_shape[axis]

        if all(i == split[0] for i in split):
            _, dtype = graph.tensor_info(node.output[0])
            output_shape = input_shape
            if keepdims or (split[0] > 1):
                output_shape[axis] //= split[0]
            else:
                output_shape.pop(axis)
            debug(f"SplitToSequence: set output shape to {output_shape}")
            graph.set_seqeuence_info(node.output[0], output_shape, dtype)
            return
        warning("[TODO] if you reach this line, SplitToSequence can't be fixed.")


@INFERSHAPE_PATCH.register()
class InferGroupNormalization(Rewriter):
    """[BUG] inference shape failed after GroupNormalization."""

    def __init__(self):
        super().__init__(SingleNodePattern("GroupNormalization"))

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        input_shape, dtype = graph.tensor_info(node.input[0])
        output_shape, _ = graph.tensor_info(node.output[0])
        if output_shape is None and input_shape is not None:
            debug(f"GroupNormalization: set output shape to {input_shape}")
            graph.set_value_info(node.output[0], input_shape, dtype)


@PASSES.register()
def infer_shape(
    graph: OnnxGraph,
    input_shapes: Optional[Dict[str, List[int | str]]] = None,
    output_shapes: Optional[Dict[str, List[int | str]]] = None,
) -> OnnxGraph:
    """Regenerate tensor info of graph."""
    model = graph.model
    if input_shapes and output_shapes:
        model = update_inputs_outputs_dims(
            model, input_dims=input_shapes, output_dims=output_shapes
        )
    model = shape_inference.infer_shapes(model, data_prop=True)
    graph = OnnxGraph(model, base_dir=graph.external_base)
    # Patch for infer_shapes bugs
    pm = get_pass_manager(list(INFERSHAPE_PATCH))
    graph = pm.optimize(graph)
    need_second_infer = False
    if sum(i.num_rewrites for i in pm.activated) > 0:
        need_second_infer = True
    if need_second_infer:
        model = graph.model
        model = shape_inference.infer_shapes(model, data_prop=True)
        return OnnxGraph(model, base_dir=graph.external_base)
    return graph
