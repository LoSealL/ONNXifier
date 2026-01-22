"""
Copyright (C) 2024-2025 The ONNXIFIER Authors.

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
from openvino.opset1 import parameter
from openvino.opset3 import broadcast

from . import convert_xml


def test_broadcast_opset3():
    p = parameter([8, 1, 1], name="data")
    b = broadcast(p, [1, 8, 10, 10])
    model = Model([b], [p])
    convert_xml(model)


def test_broadcast_bidirectional_opset3():
    p = parameter([8, 1, 1], name="data")
    b = broadcast(p, [1, 1, 10, 10], broadcast_spec="BIDIRECTIONAL")
    model = Model([b], [p])
    convert_xml(model)


def test_broadcast_expand_opset3():
    p = parameter([8, 1, 1], name="data")
    q = parameter([4], dtype=np.int64)
    b = broadcast(p, q)
    model = Model([b], [p, q])
    convert_xml(model)


def test_broadcast_axes_mapping_case1_opset3():
    p = parameter([16], name="data")
    b = broadcast(p, [1, 16, 50, 50], axes_mapping=[1], broadcast_spec="EXPLICIT")
    model = Model([b], [p])
    convert_xml(model)


def test_broadcast_axes_mapping_case2_opset3():
    p = parameter([50, 50], name="data")
    b = broadcast(p, [1, 50, 50, 16], axes_mapping=[1, 2], broadcast_spec="EXPLICIT")
    model = Model([b], [p])
    convert_xml(model)


def test_broadcast_axes_mapping_case3_opset3():
    p = parameter([50, 50], name="data")
    b = broadcast(p, [1, 50, 16, 50], axes_mapping=[1, 3], broadcast_spec="EXPLICIT")
    model = Model([b], [p])
    convert_xml(model)
