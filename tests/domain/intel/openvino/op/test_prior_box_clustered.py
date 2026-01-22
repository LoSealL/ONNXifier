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
from openvino.opset1 import constant, prior_box_clustered

from . import convert_xml


def test_prior_box_clustered_opset1():
    output_size = constant(np.array([15, 20], np.int64))
    image_size = constant(np.array([480, 640], np.int64))
    pbc_attrs = dict(
        width=[8.0213, 21.4187, 12.544, 29.6107],
        height=[12.8, 33.792, 21.76, 53.9307],
        clip=False,
        step=16,
        step_w=0,
        step_h=0,
        offset=0.5,
        variance=[0.1, 0.1, 0.2, 0.2],
    )
    pbc = prior_box_clustered(output_size, image_size, pbc_attrs)
    model = Model([pbc], [])
    convert_xml(model)
