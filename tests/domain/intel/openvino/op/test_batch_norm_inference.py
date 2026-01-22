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
from openvino.opset5 import batch_norm_inference

from . import convert_xml


def test_batch_norm_inference_opset5():
    x = parameter([2, 8, 64], name="x")
    scale = constant(np.random.uniform(0, 1, size=[8]).astype(np.float32))
    bias = constant(np.random.normal(size=[8]).astype(np.float32))
    mean = constant(np.random.normal(size=[8]).astype(np.float32))
    var = constant(np.random.uniform(0.01, 0.1, size=[8]).astype(np.float32))
    y = batch_norm_inference(x, scale, bias, mean, var, epsilon=1e-3)
    model = Model([y], [x])
    convert_xml(model)
