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
from openvino.opset1 import constant
from openvino.opset8 import prior_box

from . import convert_xml


def test_prior_box_opset8():
    output_size = constant(np.array([15, 20], np.int64))
    image_size = constant(np.array([480, 640], np.int64))
    pb_attrs = dict(
        min_size=[337.92],
        max_size=[403.2],
        aspect_ratio=[2.0],
        density=[],
        fixed_ratio=[],
        fixed_size=[],
        clip=False,
        flip=True,
        step=300.0,
        offset=0.5,
        variance=[0.1, 0.1, 0.2, 0.2],
        scale_all_sizes=True,
    )
    pb = prior_box(output_size, image_size, pb_attrs)
    model = Model([pb], [])
    convert_xml(model)
