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
from openvino.opset8 import slice as ov_slice

from . import convert_xml


def test_slice_opset8_with_axes():
    data = parameter([1, 32, 8, 8], np.float32)
    start = constant(np.array([4], dtype=np.int64))
    stop = constant(np.array([8], dtype=np.int64))
    step = constant(np.array([1], dtype=np.int64))
    axes = constant(np.array([1], dtype=np.int64))
    y = ov_slice(data, start, stop, step, axes)
    model = Model([y], [data])
    convert_xml(model)


def test_slice_opset8_without_axes():
    data = parameter([1, 32, 8, 8], np.float32)
    start = constant(np.array([4], dtype=np.int64))
    stop = constant(np.array([8], dtype=np.int64))
    step = constant(np.array([1], dtype=np.int64))
    y = ov_slice(data, start, stop, step)
    model = Model([y], [data])
    convert_xml(model)
