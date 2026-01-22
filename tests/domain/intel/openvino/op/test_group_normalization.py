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
from openvino.opset12 import group_normalization

from . import convert_xml


def test_group_normalization_opset12():
    p = parameter([1, 8, 4, 4])
    scales = constant(np.random.uniform(0, 1, size=[8]).astype(np.float32))
    bias = constant(np.random.uniform(-1, 1, [8]).astype(np.float32))
    g = group_normalization(p, scales, bias, num_groups=2, epsilon=1e-5)
    model = Model([g], [p])
    convert_xml(model, opset_version=21)
