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
from onnx import numpy_helper
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier import ONNXIFIER_OPSET, PassManager
from onnxifier.graph import OnnxGraph


def _build_model():
    const = make_node(
        "Constant",
        [],
        ["const_path"],
        name="const",
        value=numpy_helper.from_array(np.array([1.0], dtype=np.float32)),
    )
    add = make_node("Add", ["x", "const_path"], ["sum0"], name="add")
    relu = make_node("Relu", ["sum0"], ["y"], name="relu")
    graph = make_graph(
        [const, add, relu],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1]))],
        [make_value_info("y", make_tensor_type_proto(1, [1]))],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def test_debug_node_default_prefix_and_regex_match():
    graph = OnnxGraph(_build_model())
    pm = PassManager(
        ["debug_node"],
        configs={"debug_node": {"node_types": "Ad.*,Relu"}},
    )

    graph = pm.optimize(graph, strict=True)

    output_names = set(graph.outputs)
    # x is graph input, so skip it; sum0 is intermediate output
    assert output_names == {
        "y",
        "debug/add/output0",
        "debug/relu/input0",
    }
    assert "const_path" not in output_names
    assert "debug/add/input0" not in output_names  # x is graph input
    identity_outputs = {
        graph.nodes[name]["pb"].output[0]
        for name in graph.nodes
        if graph.nodes[name]["pb"].op_type == "Identity"
    }
    assert identity_outputs == {"debug/add/output0", "debug/relu/input0"}


def test_debug_node_custom_prefix():
    graph = OnnxGraph(_build_model())
    pm = PassManager(
        ["debug_node"],
        configs={"debug_node": {"node_types": "Add", "prefix": "probe/"}},
    )

    graph = pm.optimize(graph, strict=True)

    # x is graph input, so only output0 is exported
    assert set(graph.outputs) == {"y", "probe/add/output0"}
