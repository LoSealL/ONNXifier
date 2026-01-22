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

import tempfile
from pathlib import Path

import onnx

from onnxifier.domain.intel.openvino.ir.ir11 import ir_to_onnx
from onnxifier.domain.intel.openvino.xml_frontend import openvino_xml_to_onnx_graph
from onnxifier.graph import OnnxGraph

DIR = Path(__file__).parent.expanduser()


def test_srcnn_to_onnx():
    model = ir_to_onnx(DIR / "srcnn.xml", DIR / "srcnn.bin")
    model = OnnxGraph(model).model
    with tempfile.NamedTemporaryFile("wb", suffix=".onnx", delete=True) as f:
        # for debug
        onnx.save_model(model, f, "protobuf")
        print(f"save to {f.name}")


def test_srcnn_to_onnx_without_explicit_bin_file():
    ir_to_onnx(DIR / "srcnn.xml")


def test_xml_frontend():
    openvino_xml_to_onnx_graph(DIR / "srcnn.xml")
