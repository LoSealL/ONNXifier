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
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_type_proto,
    make_value_info,
)
from onnx.reference import ReferenceEvaluator

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, PassManager
from onnxifier.graph import OnnxGraph


def _build_graph():
    reduce_max = make_node(
        "ReduceMax", ["x", "max_axes"], ["max1"], "reduce_max", keepdims=0
    )
    topk = make_node(
        "TopK",
        ["max1", "topk_k"],
        ["topk1", "topk2"],
        "topk",
        axis=-1,
        largest=1,
        sorted=1,
    )
    unsqueeze = make_node(
        "Unsqueeze", ["topk2", "unsqueeze_axes"], ["unsqueeze1"], "unsqueeze"
    )
    tile = make_node("Tile", ["unsqueeze1", "repeats"], ["tile1"], "tile")
    gather_elements = make_node("GatherElements", ["x", "tile1"], ["y"], axis=1)

    graph = make_graph(
        [reduce_max, topk, unsqueeze, tile, gather_elements],
        "graph",
        # shape[1] of x can't be larger than 32, it will cause a error about np.choose
        # related link:
        # https://stackoverflow.com/questions/39162235/alternative-for-numpy-choose-that-allows-an-arbitrary-or-at-least-more-than-32-a
        [make_value_info("x", make_tensor_type_proto(1, [1, 20, 4]))],
        [make_value_info("y", make_tensor_type_proto(1, [1, 3, 4]))],
        [
            make_tensor("max_axes", 7, [1], np.array([-1], dtype="int64")),
            make_tensor("topk_k", 7, [1], np.array([3], dtype="int64")),
            make_tensor("unsqueeze_axes", 7, [1], np.array([-1], dtype="int64")),
            make_tensor("repeats", 7, [3], np.array([1, 1, 4], dtype="int64")),
        ],
    )
    return make_model(
        graph, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )


def test_gatherelements_to_gathernd():
    model = _build_graph()
    runner1 = ReferenceEvaluator(model)
    x = np.random.uniform(0, 1, size=[1, 20, 4]).astype(np.float32)
    y1 = runner1.run(None, {"x": x})[0]

    graph = OnnxGraph(model)
    pm = PassManager(["infer_shape", "gatherelements_to_gathernd", "onnxsim"])
    graph = pm.optimize(graph, strict=True)

    # check output
    runner2 = ReferenceEvaluator(graph.model)
    y2 = runner2.run(None, {"x": x})[0]
    assert np.allclose(y1, y2)

    # check conv number
    gather_elements_number = 0
    gatherND_number = 0
    for name in graph:
        if graph.nodes[name]["pb"].op_type == "GatherElements":
            gather_elements_number += 1
        elif graph.nodes[name]["pb"].op_type == "GatherND":
            gatherND_number += 1
    assert gather_elements_number == 0
    assert gatherND_number == 1
