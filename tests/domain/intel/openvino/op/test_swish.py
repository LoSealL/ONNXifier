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
from openvino.opset1 import parameter
from openvino.opset4 import swish

from . import convert_xml


def test_swish_opset4():
    p = parameter([1, 32, 8, 8])
    r = swish(p)
    m = Model([r], [p])
    convert_xml(m)


def test_swish_beta_opset4():
    p = parameter([1, 32, 8, 8])
    r = swish(p, beta=np.array(0.5, dtype=np.float32))
    m = Model([r], [p])
    convert_xml(m)
