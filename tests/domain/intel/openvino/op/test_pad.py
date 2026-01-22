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
from openvino.opset1 import constant, parameter
from openvino.opset12 import pad

from . import convert_xml


def test_pad_opset12():
    a = parameter([1, 3, 16, 16])
    beg = constant(np.array([0, 0, 1, 1], np.int64))
    end = constant(np.array([0, 0, 2, 2], np.int64))
    c = pad(a, beg, end, "constant")
    model = Model([c], [a])
    convert_xml(model)
