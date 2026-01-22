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
from openvino.opset1 import parameter, reshape

from . import convert_xml


def test_reshape_with_negative_dim_opset1():
    p = parameter(shape=[1, 3, 224, 224])
    s = reshape(p, [1, -1, 224 * 224], False)
    m = Model([s], [p])
    convert_xml(m)


def test_reshape_with_zero_dim_opset1():
    p = parameter(shape=[1, 8])
    s = reshape(p, [0, 8], True)
    m = Model([s], [p])
    convert_xml(m)
