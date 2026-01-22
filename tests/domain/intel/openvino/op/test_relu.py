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
from openvino.opset1 import constant, parameter, prelu, relu, selu
from openvino.opset7 import gelu

from . import convert_xml


def test_relu_opset1():
    p = parameter([1, 32, 8, 8])
    r = relu(p)
    model = Model([r], [p])
    convert_xml(model)


def test_prelu_opset1():
    p = parameter([1, 32, 8, 8])
    alpha = constant(np.random.rand(32).astype(np.float32))
    r = prelu(p, alpha)
    model = Model([r], [p])
    convert_xml(model)


def test_selu_opset1():
    p = parameter([1, 32, 8, 8])
    r = selu(p, np.array(1.67, np.float32), np.array(1.0507, np.float32))
    model = Model([r], [p])
    convert_xml(model)


def test_gelu_opset7():
    p = parameter([1, 32, 8, 8])
    r = gelu(p, "tanh")
    model = Model([r], [p])
    convert_xml(model)
