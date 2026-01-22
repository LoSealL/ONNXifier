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
from openvino.opset1 import constant, convolution, group_convolution, parameter

from . import convert_xml


def test_convolution_1d_opset1():
    p = parameter([1, 3, 224])
    w = constant(np.random.rand(8, 3, 7).astype(np.float32))
    conv = convolution(p, w, [1], [0], [0], [], name="conv")
    model = Model([conv], [p])
    convert_xml(model)


def test_convolution_2d_opset1():
    p = parameter([1, 3, 224, 224])
    w = constant(np.random.rand(8, 3, 7, 7).astype(np.float32))
    conv = convolution(p, w, [1, 1], [0, 0], [0, 0], [], name="conv")
    model = Model([conv], [p])
    convert_xml(model)


def test_convolution_3d_opset1():
    p = parameter([1, 3, 224, 224, 224])
    w = constant(np.random.rand(8, 3, 7, 7, 7).astype(np.float32))
    conv = convolution(p, w, [1, 1, 1], [0, 0, 0], [0, 0, 0], [], name="conv")
    model = Model([conv], [p])
    convert_xml(model)


def test_group_convolution_1d_opset1():
    p = parameter([1, 8, 224])
    w = constant(np.random.rand(8, 1, 1, 7).astype(np.float32))
    conv = group_convolution(p, w, [1], [0], [0], [1], name="conv")
    model = Model([conv], [p])
    convert_xml(model)


def test_group_convolution_2d_opset1():
    p = parameter([1, 8, 224, 224])
    w = constant(np.random.rand(8, 1, 1, 7, 7).astype(np.float32))
    conv = group_convolution(p, w, [1, 1], [0, 0], [0, 0], [1, 1], name="conv")
    model = Model([conv], [p])
    convert_xml(model)


def test_group_convolution_3d_opset1():
    p = parameter([1, 8, 224, 224, 224])
    w = constant(np.random.rand(8, 1, 1, 7, 7, 7).astype(np.float32))
    conv = group_convolution(
        p, w, [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], name="conv"
    )
    model = Model([conv], [p])
    convert_xml(model)
