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

# pylint: disable=missing-function-docstring

import numpy as np
from openvino import Model
from openvino.opset1 import parameter, squeeze

from . import convert_xml


def test_squeeze_opset1():
    p = parameter([1, 1, 2, 1, 1])
    s = squeeze(p, [])
    m = Model([s], [p])
    convert_xml(m)


def test_squeeze_cast_non_int64_axes_vector():
    p = parameter([1, 1, 2, 1, 1])
    axes = parameter([1], dtype=np.int32, name="axes")
    s = squeeze(p, axes)
    m = Model([s], [p, axes])

    model = convert_xml(m)

    op_types = [node.op_type for node in model.graph.node]
    assert "Cast" in op_types
    assert "Unsqueeze" not in op_types


def test_squeeze_unsqueeze_and_cast_scalar_axes():
    p = parameter([1, 1, 2, 1, 1])
    axes = parameter([], dtype=np.int32, name="axes")
    s = squeeze(p, axes)
    m = Model([s], [p, axes])

    model = convert_xml(m)

    op_types = [node.op_type for node in model.graph.node]
    assert "Unsqueeze" in op_types
    assert "Cast" in op_types
