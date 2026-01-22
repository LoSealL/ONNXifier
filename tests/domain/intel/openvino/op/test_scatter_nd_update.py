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
from openvino.opset4 import scatter_nd_update

from . import convert_xml


def test_scatter_nd_update_opset4():
    data = parameter([8])
    indices = constant(np.array([[4], [3], [1], [7]], np.int64))
    updates = constant(np.array([9, 10, 11, 12], np.float32))
    r = scatter_nd_update(data, indices, updates)
    m = Model([r], [data])
    convert_xml(m)
