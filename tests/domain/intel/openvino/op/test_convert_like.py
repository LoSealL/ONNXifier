"""
Copyright (C) 2026 The ONNXIFIER Authors.

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
from openvino.opset1 import convert_like, parameter

from . import convert_xml


def test_convert_like_opset1():
    p = parameter([8], name="p", dtype=np.float32)
    q = parameter([8], name="q", dtype=np.float16)
    c = convert_like(p, q)
    model = Model([c], [p, q])
    convert_xml(model)
