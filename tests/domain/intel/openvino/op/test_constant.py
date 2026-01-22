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
from openvino.opset1 import constant

from . import convert_xml


def test_concat_opset1():
    p0 = constant(np.random.randn(32).astype(np.float32), name="p0")
    p1 = constant(np.random.randn(32).astype(np.float16), name="p1")
    p2 = constant(np.random.randn(32).astype(np.int8), name="p2")
    p3 = constant(np.random.randn(32).astype(np.int64), name="p3")
    p4 = constant(np.random.randn(32).astype(np.uint8), name="p4")
    model = Model([p0, p1, p2, p3, p4], [])
    convert_xml(model)
