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
from openvino.opset13 import scaled_dot_product_attention

from . import convert_xml


def test_scaled_dot_product_attention():
    q = parameter([1, 16, 24, 64])
    k = parameter([1, 16, 32, 64])
    v = parameter([1, 16, 32, 56])
    y = scaled_dot_product_attention(q, k, v)
    model = Model(y, [q, k, v])

    convert_xml(model)


def test_scaled_dot_product_attention_attention_mask():
    q = parameter([1, 16, 24, 64])
    k = parameter([1, 16, 32, 64])
    v = parameter([1, 16, 32, 56])
    mask = constant(np.ones([1, 24, 32], np.float32))
    y = scaled_dot_product_attention(q, k, v, mask)
    model = Model(y, [q, k, v])

    convert_xml(model)


def test_scaled_dot_product_attention_causal():
    q = parameter([1, 16, 24, 64])
    k = parameter([1, 16, 32, 64])
    v = parameter([1, 16, 32, 56])
    y = scaled_dot_product_attention(q, k, v, causal=True)
    model = Model(y, [q, k, v])

    convert_xml(model)
