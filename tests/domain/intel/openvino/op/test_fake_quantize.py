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
from openvino.opset1 import constant, fake_quantize, parameter

from . import convert_xml


def test_fake_quantize_opset1():
    p = parameter([1, 8, 4, 4])
    ilow = constant(np.array([0.0], dtype=np.float32))
    ihigh = constant(np.array([1.0], dtype=np.float32))
    olow = constant(np.array([0.0], dtype=np.float32))
    ohigh = constant(np.array([255.0], dtype=np.float32))
    fq = fake_quantize(p, ilow, ihigh, olow, ohigh, 256)
    model = Model([fq], [p])
    convert_xml(model)
