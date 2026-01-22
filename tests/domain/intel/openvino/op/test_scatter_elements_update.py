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
from openvino.opset3 import scatter_update
from openvino.opset12 import scatter_elements_update

from . import convert_xml


def test_scatter_elements_update_opset12():
    axis = constant(0, dtype=np.int32)
    data = parameter([3, 3])
    indices = parameter([2, 3], dtype=np.int64)
    updates = parameter([2, 3])
    r = scatter_elements_update(data, indices, updates, axis)
    m = Model([r], [data, indices, updates])
    convert_xml(m)


def test_scatter_update_opset3():
    axis = constant(1, dtype=np.int32)
    data = parameter([3, 5])
    indices = constant(np.array([0, 2], np.int64))
    updates = parameter([3, 2])
    r = scatter_update(data, indices, updates, axis)
    m = Model([r], [data, updates])
    convert_xml(m)


def test_scatter_update_indices_rank2_opset3():
    axis = constant(0, dtype=np.int32)
    data = parameter([5, 3])
    indices = parameter([2, 2], dtype=np.int64)
    updates = parameter([2, 2, 3])
    r = scatter_update(data, indices, updates, axis)
    m = Model([r], [data, indices, updates])
    convert_xml(m)
