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

from onnxifier.graph import OnnxGraph
from onnxifier.passes import PASSES


def _build_test_graph():
    resize = make_node(
        "Resize",
        inputs=["x", "roi", "scales", "sizes"],
        outputs=["y"],
        name="resize",
        coordinate_transformation_mode="half_pixel",
        mode="cubic",
    )
    graph = make_graph(
        [resize],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 8, 8]))],
        [make_value_info("y", make_tensor_type_proto(1, None))],
        [make_tensor("scales", 1, [4], np.array([1, 1, 2, 2], "float32"))],
    )
    return make_model(graph)


def test_rewriter():
    graph = OnnxGraph(_build_test_graph())
    rewriter = PASSES.get("resize_to_nearest_neighbor")
    graph = rewriter(graph)
    assert "nearest" == rewriter.get_attribute(graph.nodes["resize"]["pb"], "mode")
    assert "asymmetric" == rewriter.get_attribute(
        graph.nodes["resize"]["pb"], "coordinate_transformation_mode"
    )
