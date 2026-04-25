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

# pylint: disable=missing-function-docstring

import onnx
import pooch
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier import ONNXIFIER_OPSET, convert, convert_graph


def test_convert_api(classification_models):
    model_url, hash_value = classification_models
    model_file = pooch.retrieve(model_url, hash_value)
    model = onnx.load_model(model_file)
    convert(model)
    convert(model_file, strict=True)


def _build_graph_with_named_identities():
    branch0_identity = make_node("Identity", ["x0"], ["y0"], "branch[0]/id")
    branch0_relu = make_node("Relu", ["y0"], ["z0"], "branch[0]/relu")
    branch1_identity = make_node("Identity", ["x1"], ["y1"], "branch[1]/id")
    branch1_relu = make_node("Relu", ["y1"], ["z1"], "branch[1]/relu")
    graph = make_graph(
        [branch0_identity, branch0_relu, branch1_identity, branch1_relu],
        "graph",
        [
            make_value_info("x0", make_tensor_type_proto(1, [1, 3, 8, 8])),
            make_value_info("x1", make_tensor_type_proto(1, [1, 3, 8, 8])),
        ],
        [
            make_value_info("z0", make_tensor_type_proto(1, [1, 3, 8, 8])),
            make_value_info("z1", make_tensor_type_proto(1, [1, 3, 8, 8])),
        ],
        [],
    )
    return make_model(graph, opset_imports=[ONNXIFIER_OPSET])


def test_convert_graph_api_specify_node_names_exact_match():
    graph = convert_graph(
        _build_graph_with_named_identities(),
        passes=["eliminate_identity"],
        strict=True,
        print_passes=False,
        specify_node_names=["branch[0]/id"],
    )

    assert len(graph.nodes) == 3
    assert "branch[0]/id" not in graph
    assert "branch[1]/id" in graph


def test_convert_graph_api_specify_node_names_regex_match():
    graph = convert_graph(
        _build_graph_with_named_identities(),
        passes=["eliminate_identity"],
        strict=True,
        print_passes=False,
        specify_node_names=[r"branch\[[01]\]/id"],
    )

    assert len(graph.nodes) == 2
    assert "branch[0]/id" not in graph
    assert "branch[1]/id" not in graph
