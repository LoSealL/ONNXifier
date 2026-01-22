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
from openvino.opset1 import parameter
from openvino.opset14 import avg_pool

from . import convert_xml


def test_avgpool_opset14():
    p = parameter([1, 3, 224, 224], name="data")
    pool = avg_pool(p, [2, 2], [0, 0], [0, 0], [2, 2], True, name="pool")
    model = Model([pool], [p])
    convert_xml(model)


def test_avgpool_auto_pad_opset14():
    p = parameter([1, 3, 53, 53], name="data")
    pool = avg_pool(p, [1, 1], [0, 0], [0, 0], [3, 3], True, auto_pad="same_upper")
    model = Model([pool], [p])
    convert_xml(model)
