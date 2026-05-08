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

import os
from pathlib import Path

import onnx

from ..logger import warning
from .intel import IR_DOMAIN
from .trt import TRT_IR_DOMAIN

_LAZY_LOAD = {}


def _make_key(operatorset: onnx.OperatorSetIdProto):
    return (operatorset.domain, operatorset.version)


def _lazy_load_xml_frontend():
    # pylint: disable=import-outside-toplevel
    from .intel.openvino.xml_frontend import (
        openvino_xml_to_onnx_graph as _openvino_xml_to_onnx_graph,
    )

    _LAZY_LOAD[_make_key(IR_DOMAIN)] = _openvino_xml_to_onnx_graph


def _lazy_register_trt_schema():
    # pylint: disable=import-outside-toplevel
    from .trt.ops import (
        attention_plugin,
        dequantize_linear,
        mamba_plugin,
        vit_attention_plugin,
    )

    if not onnx.defs.has(
        attention_plugin.attention_plugin_schema.name,
        attention_plugin.attention_plugin_schema.domain,
    ):
        onnx.defs.register_schema(attention_plugin.attention_plugin_schema)
    if not onnx.defs.has(
        vit_attention_plugin.vit_attention_plugin_schema.name,
        vit_attention_plugin.vit_attention_plugin_schema.domain,
    ):
        onnx.defs.register_schema(vit_attention_plugin.vit_attention_plugin_schema)
    if not onnx.defs.has(
        dequantize_linear.dequantize_linear_schema.name,
        dequantize_linear.dequantize_linear_schema.domain,
    ):
        onnx.defs.register_schema(dequantize_linear.dequantize_linear_schema)
    if not onnx.defs.has(
        mamba_plugin.causal_conv1d_schema.name,
        mamba_plugin.causal_conv1d_schema.domain,
    ):
        onnx.defs.register_schema(mamba_plugin.causal_conv1d_schema)


def openvino_xml_to_onnx_graph(
    xml_path: str | os.PathLike | onnx.ModelProto,
    bin_path: str | os.PathLike | None = None,
):
    """Isolate OpenVINO domain unless a IR xml is detected."""
    key = _make_key(IR_DOMAIN)
    if key in _LAZY_LOAD:
        return _LAZY_LOAD[key](xml_path, bin_path)
    raise ImportError("OpenVINO is not imported yet!")


def detect_domain(
    model: str | os.PathLike | onnx.ModelProto,
) -> list[onnx.OperatorSetIdProto]:
    """Detect a custom domain of a model."""

    if isinstance(model, onnx.ModelProto):
        if len(model.opset_import) == 0:
            return []
        opsets = {}
        for opset in model.opset_import:
            if opset.domain not in ("", "ai.onnx", "ai.onnx.ml"):
                opsets[opset.domain] = opset
        for op in model.graph.node:
            if op.domain not in ("", "ai.onnx", "ai.onnx.ml"):
                if op.domain not in opsets:
                    warning("Found op with domain '%s' without import!", op.domain)
                    opsets[op.domain] = onnx.helper.make_operatorsetid(op.domain, 1)
        for domain in opsets:
            if domain == IR_DOMAIN.domain:
                _lazy_load_xml_frontend()
            if domain == TRT_IR_DOMAIN.domain:
                _lazy_register_trt_schema()
        return list(opsets.values())
    model = Path(model).resolve()
    if model.suffix.lower() == ".xml":
        _lazy_load_xml_frontend()
        return [IR_DOMAIN]
    return []


__all__ = ["openvino_xml_to_onnx_graph", "detect_domain"]
