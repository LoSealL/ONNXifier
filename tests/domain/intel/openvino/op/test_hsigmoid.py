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
from openvino.opset1 import hard_sigmoid, parameter
from openvino.opset5 import hsigmoid

from . import convert_xml


def test_hsigmoid_opset5():
    p = parameter([1, 8, 4, 4])
    a = hsigmoid(p)
    model = Model([a], [p])
    convert_xml(model)


def test_hard_sigmoid_opset1():
    p = parameter([1, 8, 4, 4])
    a = hard_sigmoid(p, alpha=np.float32(0.2), beta=np.float32(0.3))
    model = Model([a], [p])
    convert_xml(model)
