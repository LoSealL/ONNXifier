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
from openvino.opset1 import constant, normalize_l2, parameter

from . import convert_xml


def test_normalize_l2_axes_single_ints_opset1():
    p = parameter([1, 3, 224, 224])
    axes = constant([1], np.int64)
    r = normalize_l2(p, axes, 1e-5, "add")
    model = Model([r], [p])

    convert_xml(model)


def test_normalize_l2_axes_int_opset1():
    p = parameter([1, 3, 224, 224])
    axes = constant(1, np.int64)
    r = normalize_l2(p, axes, 1e-5, "add")
    model = Model([r], [p])

    convert_xml(model)


def test_normalize_l2_axes_ints_opset1():
    p = parameter([1, 3, 224, 224])
    axes = constant([2, 3], np.int64)
    r = normalize_l2(p, axes, 1e-5, "add")
    model = Model([r], [p])

    convert_xml(model)
