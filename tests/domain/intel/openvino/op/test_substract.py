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

from openvino import Model
from openvino.opset1 import parameter, subtract

from . import convert_xml


def test_substract_opset1():
    p1 = parameter([1, 64, 8, 8])
    p2 = parameter([1, 64, 1, 1])
    s = subtract(p1, p2)
    m = Model([s], [p1, p2])
    convert_xml(m)
