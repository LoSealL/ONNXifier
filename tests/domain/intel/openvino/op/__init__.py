"""
Copyright (C) 2026 The ONNXIFIER Authors.

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

import tempfile
from pathlib import Path

import onnx
from openvino import serialize

from onnxifier.domain.intel.openvino.xml_frontend import openvino_xml_to_onnx_graph


def convert_xml(model, opset_version=None):
    """test model conversion from openvino xml to onnx"""
    with tempfile.TemporaryDirectory() as tmpdir:
        serialize(model, Path(tmpdir) / "model.xml", Path(tmpdir) / "model.bin")
        model = openvino_xml_to_onnx_graph(Path(tmpdir) / "model.xml")
        if opset_version is not None:
            model.opset_import[0].domain = ""
            model.opset_import[0].version = opset_version
        onnx.checker.check_model(model, True)
        onnx.save_model(model, Path(tmpdir) / "model.onnx")
    return model
