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

from openvino import Model
from openvino.opset1 import parameter, select

from . import convert_xml


def test_select_opset1():
    p = parameter([8])
    q = parameter([8])
    c = parameter([1], dtype=bool)
    r = select(c, p, q)
    m = Model([r], [p, q, c])
    convert_xml(m)
