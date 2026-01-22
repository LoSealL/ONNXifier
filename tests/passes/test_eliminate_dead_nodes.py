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

from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, PassManager
from onnxifier.graph import OnnxGraph


def _build_graph():
    a = make_node("Relu", ["a"], ["a0"])
    b = make_node("Relu", ["b"], ["b0"])
    g = make_graph(
        [a, b],
        "test",
        [
            make_tensor_value_info("a", 1, [32]),
            make_tensor_value_info("b", 1, [32]),
        ],
        [
            make_tensor_value_info("a0", 1, [32]),
        ],
    )
    return make_model(
        g, ir_version=ONNXIFIER_IR_VERSION, opset_imports=[ONNXIFIER_OPSET]
    )


def test_eliminate_dead_nodes():
    model = _build_graph()
    graph = OnnxGraph(model)
    pm = PassManager(["eliminate_dead_nodes"])
    graph = pm.optimize(graph, strict=True)

    assert len(graph) == 1
