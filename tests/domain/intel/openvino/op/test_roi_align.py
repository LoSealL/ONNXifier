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
from openvino.opset9 import roi_align

from . import convert_xml


def test_roi_align_opset9():
    p = parameter([1, 32, 128, 128])
    roi_xy = np.random.uniform(0, 128, size=[100, 2])
    roi_wh = np.random.uniform(1, 128, size=[100, 2])
    rois = np.concatenate((roi_xy, roi_xy + roi_wh), axis=-1)
    rois = np.clip(rois, 0, 128)
    rois = constant(rois.astype(np.float32))
    indices = constant(np.zeros([100], np.int64))
    ra_attrs = dict(
        mode="avg",
        pooled_h=8,
        pooled_w=8,
        sampling_ratio=1,
        spatial_scale=np.float32(1.0),
        aligned_mode="asymmetric",
    )
    ra = roi_align(p, rois, indices, **ra_attrs)
    model = Model([ra], [p])
    convert_xml(model)
