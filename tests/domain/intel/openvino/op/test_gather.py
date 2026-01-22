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
from openvino.opset1 import constant, gather, parameter

from . import convert_xml


def test_gather_opset1():
    p = parameter([1, 8, 4, 4])
    indice = constant(np.array([0, 2, 4, 6], dtype=np.int64))
    axis = constant(np.array(1, dtype=np.int64))
    g = gather(p, indice, axis)
    model = Model([g], [p])
    convert_xml(model)
