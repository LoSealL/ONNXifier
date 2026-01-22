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

# pylint: disable=missing-function-docstring

import numpy as np
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier import PassManager
from onnxifier.graph import OnnxGraph


def _build_test_graph(slope_value):
    prelu = make_node(
        "PRelu",
        inputs=["x", "slope"],
        outputs=["y"],
        name="prelu",
    )
    graph = make_graph(
        [prelu],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 32, 8, 8]))],
        [make_value_info("y", make_tensor_type_proto(1, [1, 32, 8, 8]))],
        [make_tensor("slope", 1, slope_value.shape, slope_value)],
    )
    return make_model(graph)


def test_prelu_rewriter():
    slope = np.array([0.2], "float32")
    graph = OnnxGraph(_build_test_graph(slope))
    pm = PassManager(["prelu_to_leaky"])
    graph = pm.optimize(graph, strict=True)
    assert graph.nodes["prelu/to_leaky"]["pb"].op_type == "LeakyRelu"


def test_prelu_rewriter_failure_case():
    slope = np.array([0.2, 0.3, 0.4, 0.5], "float32")
    graph = OnnxGraph(_build_test_graph(slope))
    pm = PassManager(["initializer_to_constant", "prelu_to_leaky"])
    graph = pm.optimize(graph, strict=True)
    assert graph.nodes["prelu"]["pb"].op_type == "PRelu"
