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

# pylint: disable=arguments-differ

from collections import defaultdict
from typing import Dict, List

import networkx as nx
import numpy as np
from onnx import numpy_helper
from onnx.helper import make_node
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import PASSES
from ..pattern import GraphPattern, SingleNodePattern, StartEndPointPattern
from ..rewriter import Rewriter
from ..utils import make_constant


@PASSES.register(deps=["eliminate_dead_nodes", "initializer_to_constant"])
def yolov5_5d_to_4d(graph: OnnxGraph) -> OnnxGraph:  # noqa: C901
    """Convert YOLOv5 5D subgraph to equivalent 4D subgraph."""
    node_to_add = []
    node_to_remove = []
    nodes_5d: Dict[int, List[int]] = defaultdict(list)
    for node_id in nx.topological_sort(graph):
        node_pb = graph.nodes[node_id]["pb"]
        if node_pb.op_type == "Constant":
            continue
        input_5d = []
        output_5d = []
        preds = graph.onnx_predecessors(node_pb)
        preds = [i for i in preds if i.op_type != "Constant"]
        for i in node_pb.input:
            if not i:
                continue
            ndim = len(graph.tensor_shape(i))
            if ndim == 5 and any(i in p.output for p in preds):
                input_5d.append(i)
        for i in node_pb.output:
            if len(graph.tensor_shape(i)) == 5:
                output_5d.append(i)
        if input_5d and output_5d:
            # both input and output is 5D
            if not nodes_5d:
                nodes_5d[node_id].append(node_id)
            else:
                new_head = True
                for head in nodes_5d:
                    if nx.has_path(graph, head, node_id):
                        nodes_5d[head].append(node_id)
                        new_head = False
                        break
                if new_head:
                    nodes_5d[node_id].append(node_id)
    # yolo v5 check
    if len(nodes_5d) != 3:  # 3 branches
        return graph
    for head in nodes_5d:
        head_node = graph.nodes[head]["pb"]
        # insert 4d reshape
        for i, input_name in enumerate(head_node.input):
            shape = graph.static_tensor_shape(input_name)
            shape[3] *= shape.pop(-1)  # [N, C, D, H, W] -> [N, C, D, H*W]
            shape_cst = make_constant(
                f"{head}/Reshape/Const", np.array(shape, dtype="int64")
            )
            reshape_node = make_node(
                "Reshape",
                [input_name, shape_cst.output[0]],
                [f"{head}/Reshape_Output"],
                name=f"{head}/Reshape",
            )
            head_node.input[i] = f"{head}/Reshape_Output"
            node_to_add.extend([reshape_node, shape_cst])
        for node in nodes_5d[head]:
            node_pb = graph.nodes[node]["pb"]
            if node_pb.op_type == "Transpose":
                for attr in node_pb.attribute:
                    if attr.name == "perm":
                        attr.ints.remove(4)
            elif node_pb.op_type in ("Split", "Concat"):
                for attr in node_pb.attribute:
                    if attr.name == "axis":
                        attr.i = 3
            else:
                for pred in graph.onnx_predecessors(node_pb):
                    # replace 5d constant with 4d
                    if pred.op_type == "Constant":
                        data = numpy_helper.to_array(pred.attribute[0].t)
                        shape = list(data.shape)
                        if data.ndim != 5:
                            continue
                        # [N, C, H, W, D] -> [N, C, -1, D]
                        shape = [*shape[:2], shape[2] * shape[3], shape[4]]
                        new_node = make_constant(f"{pred.name}/4D", data.reshape(shape))
                        for i, j in enumerate(node_pb.input):
                            if j == pred.output[0]:
                                node_pb.input[i] = new_node.output[0]
                        node_to_remove.append(pred)
                        node_to_add.append(new_node)

    for node in node_to_remove:
        graph.remove_onnx_node(node)
    for node in node_to_add:
        graph.add_onnx_node(node)
    return graph


@PASSES.register("plate_5d_to_4d")
class Plate5DTo4DRewriter(Rewriter):
    """Rewrite YOLOv5 (Plate Detection) 5D subgraph to equivalent 4D subgraph."""

    def __init__(self):
        gstart = SingleNodePattern("Reshape").with_name("/model.21/Reshape")
        gstart |= SingleNodePattern("Reshape").with_name("/model.21/Reshape_2")
        gstart |= SingleNodePattern("Reshape").with_name("/model.21/Reshape_4")
        gend = GraphPattern()
        r = SingleNodePattern("Reshape")
        gend.add_edge(SingleNodePattern("Concat"), r)
        gend.add_edge(r, SingleNodePattern("Concat"))
        pattern = StartEndPointPattern(
            start_pattern=gstart,
            end_pattern=gend,
        )
        super().__init__(pattern)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        reshapes = [i for i in nodes if i.op_type == "Reshape"]
        assert len(reshapes) == 2, f"{len(reshapes)} reshapes found, expected 2"
        if not nx.has_path(graph, reshapes[0].name, reshapes[1].name):
            reshapes = [reshapes[1], reshapes[0]]
        assert nx.has_path(graph, reshapes[0].name, reshapes[1].name)
        shape1 = self.get_value_or_die(reshapes[0].input[1])
        new_shape1 = make_constant(
            f"{reshapes[0].name}/reshape",
            np.array([*shape1[:3], shape1[3] * shape1[4]], np.int64),
        )
        reshapes[0].input[1] = new_shape1.output[0]
        self += new_shape1

        transposes = [i for i in nodes if i.op_type == "Transpose"]
        assert len(transposes) == 1
        self.set_attribute(transposes[0], "perm", [0, 1, 3, 2])

        for node in nodes:
            if node.op_type == "Slice":
                axes = self.get_value_or_die(node.input[3])
                assert axes[0] == 4
                new_axes = make_constant(f"{node.name}/axes", np.array([3], np.int64))
                node.input[3] = new_axes.output[0]
                self += new_axes

        concats = [i for i in nodes if i.op_type == "Concat"]
        assert len(concats) == 3
        for node in concats:
            axis = self.get_attribute(node, "axis")
            if axis == 1:
                continue
            self.set_attribute(node, "axis", 3)

        for node in nodes:
            if node.op_type not in ("Mul", "Add"):
                continue
            operand = self.get_value(node.input[1])
            assert operand is not None
            if operand.ndim == 5:
                shape = operand.shape
                operand_node = make_constant(
                    f"{node.name}/operand", operand.reshape([*shape[:2], -1, shape[-1]])
                )
                node.input[1] = operand_node.output[0]
                self += operand_node
