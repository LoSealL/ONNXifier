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

import os
from typing import Optional

import onnx

from .... import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET, PassManager
from ....graph import OnnxGraph
from .ir.ir11 import ir_to_onnx
from .passes.op import IR_PASSES, OP_CONVERT

_POST_PASSES = [
    "conv_dequantize_weight",
    "gemm_dequantize_weight",
    "recalculate_dequantize_weight",
    "eliminate_dead_nodes",
]


def openvino_xml_to_onnx_graph(
    model_path: str | os.PathLike | onnx.ModelProto,
    model_bin: Optional[str | os.PathLike] = None,
) -> onnx.ModelProto:
    """Convert OpenVINO IR .xml file to an equivalent ONNX graph.

    Args:
        model_path (str | Path): Path to the OpenVINO IR.xml file.
        model_bin (str | Path, optional): Path to the OpenVINO IR.bin file.
            Defaults to None. If model_bin is None, the binary is to be found in the
            same directory as the xml file with the same name but with a.bin extension.
    """
    if isinstance(model_path, onnx.ModelProto) and model_bin is None:
        ov_onnx = model_path  # for nested call but do not expose this interface
    elif not isinstance(model_path, onnx.ModelProto):
        ov_onnx = ir_to_onnx(model_path, model_bin)
    else:
        raise ValueError("model_bin is specified but model_path is ModelProto.")
    graph = OnnxGraph(ov_onnx)
    # sort OP_CONVERT here to make sure to put "Trivial" to the end
    passes = set(IR_PASSES).difference(OP_CONVERT)
    pm = PassManager(list(passes) + sorted(OP_CONVERT) + _POST_PASSES)
    model = pm.optimize(graph, strict=True).model
    model.ir_version = ONNXIFIER_IR_VERSION
    for opset in list(model.opset_import):
        if opset.domain not in ("", "ai.onnx", "ai.onnx.ml"):
            model.opset_import.remove(opset)
    if not model.opset_import:
        model.opset_import.append(ONNXIFIER_OPSET)
    return model
