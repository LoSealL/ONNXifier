"""
Copyright (C) 2024-2025 The ONNXIFIER Authors.

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

import onnx.version
from onnx.version_converter import convert_version

from ... import logger
from ...graph import OnnxGraph


def upgrade_op_version(graph: OnnxGraph, op_version: int = 17):
    """Upgrade the op version of all nodes in the graph to the specified version."""

    if graph.opset_version >= op_version:
        return graph

    logger.debug(f"Upgrading opset version from {graph.opset_version} to {op_version}")
    if graph.functions:
        logger.warning(
            "upgrade onnx version will lose functions in the graph, "
            f"this is a bug of onnx {onnx.version.version}. "
            f"Add -v={graph.opset_version} to avoid this auto upgradation."
        )
    new_model = convert_version(graph.model, op_version)
    return OnnxGraph(new_model, base_dir=graph.external_base)
