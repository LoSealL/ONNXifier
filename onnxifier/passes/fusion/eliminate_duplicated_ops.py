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

from typing import cast

import networkx as nx
import numpy as np
import onnx
from onnx.helper import make_node, tensor_dtype_to_np_dtype

from ...evaluator import Evaluator
from ...graph import OnnxGraph
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(name="eliminate_duplicated_transpose", deps=["infer_shape"])
class EliminateDuplicatedTranspose(Rewriter):
    """
    Eliminates monopath sequences of shape-altering operations
    (Transpose, Reshape, Squeeze, Unsqueeze) if their combined
    effect on the physical data layout is effectively a no-op
    (e.g., Transpose(1,2) -> Reshape -> Transpose(1,2)).
    """

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Transpose"))

    def rewrite(self, graph: OnnxGraph, nodes: list[onnx.NodeProto], *_a, **_kw):
        node1 = nodes[0]

        current_node = node1
        path_nodes = [node1]

        # 1. Explore subsequent monopath format-altering (shape-wise) nodes.
        if not nx.is_directed_acyclic_graph(graph):
            # Make sure graph is DAG (currently graph is not guaranteed to be DAG).
            raise RuntimeError("Graph for now is not a valid DAG!")
        while True:
            succs = graph.onnx_successors(current_node)
            # If there are multiple branches, side-effects on consumers exist,
            # so terminate exploration to keep the path pure.
            if len(succs) != 1:
                break
            succ = succs[0]
            if succ.op_type in ["Reshape", "Squeeze", "Unsqueeze", "Transpose"]:
                path_nodes.append(succ)
                current_node = succ
            else:
                break

        # 2. Skip if the path only contains itself
        if len(path_nodes) < 2:
            return

        final_node = path_nodes[-1]
        input_name = node1.input[0]
        output_name = final_node.output[0]

        input_shape, input_dtype = graph.tensor_info(input_name)
        output_shape = graph.tensor_shape(output_name)

        # 3. Exact static shapes for both ends are required.
        # Dynamic dimensions (None or <=0)
        # prevent us from guaranteeing exact structural alignment.
        if input_shape is None or output_shape is None:
            return
        if not all(isinstance(s, int) and s > 0 for s in input_shape):
            return
        if not all(isinstance(s, int) and s > 0 for s in output_shape):
            return
        if input_dtype == onnx.TensorProto.UNDEFINED:
            return

        input_shape = cast(list[int], input_shape)
        output_shape = cast(list[int], output_shape)

        # 4. Core validation: Use a sequential tensor `0,1,2...M-1`
        # to simulate the data layout through the ops.
        # The flattened input array represents the original linear physical layout.
        try:
            np_dtype = tensor_dtype_to_np_dtype(input_dtype)
            x = np.arange(np.prod(input_shape), dtype=np_dtype).reshape(input_shape)

            # Slice out the subgraph and use the Evaluator
            # to compute the true physical shape evolution.
            subg = graph.onnx_subgraph(path_nodes)

            evaluator = Evaluator(subg.model, backend="onnx")

            # Assemble inference feeds
            inputs_feed = {input_name: x}
            for inp in subg.input:
                if inp.name != input_name:
                    # Extract required external parameter values
                    # (e.g. Reshape's shape tensor)
                    val = self.get_value_or_die(inp.name)
                    inputs_feed[inp.name] = val

            y = evaluator([output_name], inputs_feed)[0]

            # The exact cancellation criterion:
            # If after all Transposes & Reshapes the output retains
            # the exact 0~M-1 sequential order,
            # then all dimensional shifts cancelled out.
            # The physical layout and indices remain identical.
            if not np.array_equal(np.array(y).flatten(), x.flatten()):
                return
        except Exception:  # pylint: disable=broad-except
            # Safe fallback:
            # If extracting any node dependencies fails
            # and crashes the evaluator, safely skip.
            return

        # 5. Replace subgraph: Since the physical memory layout is not disrupted at all,
        # we can simply bridge the original input and final output
        # with a single Reshape for the target shape.
        shape_name = f"{node1.name}/targetShape"
        shape_const = make_constant(
            shape_name, np.array([output_shape], dtype=np.int64)
        )
        reshape_node = make_node(
            "Reshape",
            inputs=[input_name, shape_const.output[0]],
            outputs=[output_name],
            name=f"{node1.name}/reshape",
        )

        self += [shape_const, reshape_node]

        final_succs = graph.onnx_successors(final_node)
        for succ in final_succs:
            for i, inp in enumerate(succ.input):
                if inp == output_name:
                    # self.replace is typically used, but adding new node
                    # with same output_name overrides it
                    succ.input[i] = output_name
                    self += succ

        # we can remove intermediate nodes later with eliminate_dead_nodes pass
        # or manually. For isolated path, we can just remove all:
        self -= path_nodes
