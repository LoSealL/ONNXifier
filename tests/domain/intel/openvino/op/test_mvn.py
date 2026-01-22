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

from openvino import Model
from openvino.opset1 import parameter
from openvino.opset2 import mvn as mvn_1
from openvino.opset6 import mvn as mvn_6

from . import convert_xml


def test_mvn_to_layernorm_opset1_across_channels():
    p = parameter([1, 32, 8, 8])
    r = mvn_1(p, True, True, 1e-5)
    m = Model([r], [p])

    convert_xml(m)


def test_mvn_to_layernorm_opset1_reduction_axes():
    p = parameter([1, 32, 8, 8])
    r = mvn_1(p, False, True, 1e-5)
    m = Model([r], [p])

    convert_xml(m)


def test_mvn_to_layernorm_opset6():
    p = parameter([1, 32, 8, 8])
    r = mvn_6(p, [2, 3], True, 1e-5, "inside_sqrt")
    m = Model([r], [p])

    convert_xml(m)


def test_mvn_to_layernorm_opset6_neg_axes():
    p = parameter([1, 32])
    r = mvn_6(p, [-1], True, 1e-5, "inside_sqrt")
    m = Model([r], [p])

    convert_xml(m)


def test_mvn_no_normalize_opset6():
    p = parameter([1, 32, 8, 8])
    # NOTE: OpenVINO inference is not correct when axes=[2, 3] or axes=[3]
    # There is a bug in onnx frontend that axes are translated to [1,2,3,4]
    # P.S. Onnxruntime inference is correct.
    # https://github.com/openvinotoolkit/openvino/issues/25169
    # NOTE: Fixed after 2024.3
    r = mvn_6(p, [2, 3], False, 1e-5, "inside_sqrt")
    m = Model([r], [p])

    convert_xml(m)


def test_mvn_to_multi_nodes_opset6():
    p = parameter([1, 32, 8, 8])
    r = mvn_6(p, [1, 2, 3], True, 1e-5, "outside_sqrt")
    m = Model([r], [p])

    convert_xml(m)
