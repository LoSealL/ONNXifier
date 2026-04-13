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

from pathlib import Path

from ... import OnnxGraph
from .. import PASSES


@PASSES.register("restore_external_data")
def restore_external_data(graph: OnnxGraph, external_data_dir: str = "."):
    """Restore external data for tensors stored outside the ONNX file.

    Args:
        external_data_dir (str): Directory where external data files are stored.

    Example:

        onnxifier model.onnx -a restore_external_data --external_data_dir="."
    """

    graph.external_base = external_data_dir
    graph.restore_tensors_from_external()
    return graph


@PASSES.register("save_external_data")
def save_external_data(graph: OnnxGraph, external_data: str = "."):
    """Save tensors as external data in a specified directory.

    Args:
        external_data (str): Directory or file where external data files will be saved.

    Example:

        onnxifier model.onnx -a save_external_data --external_data="."
    """

    if Path(external_data).is_dir():
        location = Path(external_data) / "model_data"
    else:
        location = Path(external_data)
    graph.save_tensors_to_external(location=location)
    return graph
