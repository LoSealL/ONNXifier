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

import numpy as np
from openvino import Model
from openvino.opset1 import constant, parameter
from openvino.opset8 import gather_nd

from . import convert_xml


def test_gather_nd_opset8():
    data = parameter([2, 2])
    indices = constant(np.array([[1], [0]], np.int64))
    r = gather_nd(data, indices, 1)
    m = Model([r], [data])
    convert_xml(m)
