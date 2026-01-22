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
from openvino.opset4 import range as ov_range

from . import convert_xml


def test_range_opset4():
    start = parameter([], np.float32)
    limit = constant(np.array(32, dtype=np.float32))
    delta = constant(np.array(1, dtype=np.float32))
    y = ov_range(start, limit, delta)
    model = Model([y], [start])
    convert_xml(model)


def test_range_opset4_int32():
    start = parameter([], np.int32)
    limit = constant(np.array(32, dtype=np.int32))
    delta = constant(np.array(1, dtype=np.int32))
    y = ov_range(start, limit, delta)
    model = Model([y], [start])
    convert_xml(model)
