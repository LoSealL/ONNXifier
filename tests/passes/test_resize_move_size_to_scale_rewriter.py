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
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor,
    make_tensor_type_proto,
    make_value_info,
)

from onnxifier.graph import OnnxGraph
from onnxifier.passes.canonicalization.resize_move_size_to_scale import (
    ResizeMoveSizeToScaleRewriter,
)
from onnxifier.passes.swap.initializer_to_constant import initializer_to_constant


def _build_test_graph():
    resize = make_node(
        "Resize",
        inputs=["x", "roi", "scales", "sizes"],
        outputs=["y"],
        name="resize",
        coordinate_transformation_mode="tf_crop_and_resize",
        mode="cubic",
        axes=[2, 3],
    )
    graph = make_graph(
        [resize],
        "graph",
        [make_value_info("x", make_tensor_type_proto(1, [1, 3, 8, 8]))],
        [make_value_info("y", make_tensor_type_proto(1, None))],
        [
            make_tensor(
                "roi", TensorProto.INT64, [4], np.array([2, 2, 6, -2], "int64")
            ),
            make_tensor("sizes", TensorProto.INT64, [2], np.array([8, 8], "int64")),
        ],
    )
    return make_model(graph)


def test_roi_rewriter():
    graph = OnnxGraph(_build_test_graph())
    rewriter = ResizeMoveSizeToScaleRewriter()
    graph = rewriter(initializer_to_constant(graph))
