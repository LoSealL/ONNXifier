"""An Open Neural Network Exchange (ONNX) Optimization and Transformation Tool.

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

__version__ = "2.2.1"

import os
import re
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Literal

import onnx
from onnx import ModelProto
from onnx.helper import make_operatorsetid

from .domain import IR_DOMAIN, detect_domain, openvino_xml_to_onnx_graph
from .graph import OnnxGraph
from .logger import debug
from .pass_manager import PassManager, print_pass_simple
from .passes.version_converter.downgrade import downgrade_op_version
from .passes.version_converter.upgrade import upgrade_op_version


def _normalize_specify_node_names(
    graph: OnnxGraph, specify_node_names: Sequence[str] | None
) -> set[str] | None:
    """Resolve exact names and regex patterns to concrete node names.

    Exact node names take priority. If an entry does not exactly match any node in the
    graph, it is treated as a regular expression and matched against node names.
    """

    if specify_node_names is None:
        return None

    graph_node_names = (
        set(graph.nodes)
        | set(graph.initializers)
        | set(graph.inputs)
        | set(graph.outputs)
    )
    resolved_node_names: set[str] = set()
    for pattern in specify_node_names:
        if pattern in graph_node_names:
            resolved_node_names.add(pattern)
            continue
        try:
            regex = re.compile(pattern)
        except re.error as ex:
            raise ValueError(f"Invalid node name regex {pattern!r}: {ex}") from ex
        resolved_node_names.update(
            node_name for node_name in graph_node_names if regex.search(node_name)
        )
    return resolved_node_names


def convert_graph(
    model: str | os.PathLike | ModelProto,
    passes: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    onnx_format: Literal["protobuf", "textproto", "json", "onnxtxt"] | None = None,
    strict: bool = False,
    configs: dict[str, Any] | None = None,
    print_passes: bool = True,
    target_opset: int | None = None,
    recursive: bool = False,
    specify_node_names: Sequence[str] | None = None,
) -> OnnxGraph:
    """Convert an ONNX model to OnnxGraph

    Args:
        model (str | os.PathLike | ModelProto): path to the model or a loaded model.
        passes (Sequence[str], optional): Names of selected passes. Defaults to None.
        exclude (Sequence[str], optional): Names of excluded passes. Defaults to None.
        onnx_format (str, optional): The serialization format of model file.
        strict (bool, optional): Break if any pass goes wrong. Defaults to False.
        configs (dict, optional): Specify configuration for passes
        print_passes (bool, optional): Print the selected passes. Defaults to True.
        target_opset (int, optional): Target opset version for ONNX domain. Defaults
            to ``ONNXIFIER_OPSET.version``.
        recursive (bool, optional): Apply passes to functions recursively.
        specify_node_names (Sequence[str], optional): Specify exact node names or
            regex patterns to apply passes only on matched nodes.

    Returns:
        OnnxGraph: converted graph
    """
    base_dir = None
    for opset in detect_domain(model):
        if opset.domain == IR_DOMAIN.domain and opset.version <= IR_DOMAIN.version:
            model = openvino_xml_to_onnx_graph(model)
    if isinstance(model, (str, os.PathLike)):
        base_dir = os.path.dirname(model)
        model = onnx.load_model(model, format=onnx_format, load_external_data=False)
        detect_domain(model)
    else:
        model = deepcopy(model)
    graph = OnnxGraph(model, base_dir=base_dir)
    if target_opset is None:
        # align models to opset v19 because all passes is designed under opset19
        graph = upgrade_op_version(graph, op_version=ONNXIFIER_OPSET.version)
    pm = PassManager(passes, exclude=exclude, configs=configs)
    if print_passes:
        print_pass_simple(pm)
    node_names = _normalize_specify_node_names(graph, specify_node_names)
    if node_names:
        debug("Filtered names: %s", node_names)
    graph = pm.optimize(
        graph,
        strict=strict,
        recursive=recursive,
        specify_node_names=node_names,
    )
    if target_opset is not None:
        if target_opset < graph.opset_version:
            graph = downgrade_op_version(graph, target_opset)
        else:
            graph = upgrade_op_version(graph, target_opset)
    return graph


def convert(
    model: str | os.PathLike | ModelProto,
    passes: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    onnx_format: Literal["protobuf", "textproto", "json", "onnxtxt"] | None = None,
    strict: bool = False,
    configs: dict[str, Any] | None = None,
    print_passes: bool = True,
    target_opset: int | None = None,
    recursive: bool = False,
    specify_node_names: Sequence[str] | None = None,
) -> ModelProto:
    """Convert an ONNX model with default or given passes

    Args:
        model (str | os.PathLike | ModelProto): path to the model or a loaded model.
        passes (Sequence[str], optional): Names of selected passes. Defaults to None.
        exclude (Sequence[str], optional): Names of excluded passes. Defaults to None.
        onnx_format (str, optional): The serialization format of model file.
        strict (bool, optional): Break if any pass goes wrong. Defaults to False.
        configs (dict, optional): Specify configuration for passes.
        print_passes (bool, optional): Print the selected passes. Defaults to True.
        target_opset (int, optional): Target opset version for ONNX domain. Defaults
            to ``ONNXIFIER_OPSET.version``.
        recursive (bool, optional): Apply passes to functions recursively.
        specify_node_names (Sequence[str], optional): Specify exact node names or
            regex patterns to apply passes only on matched nodes.
    """

    graph = convert_graph(
        model=model,
        passes=passes,
        exclude=exclude,
        onnx_format=onnx_format,
        strict=strict,
        configs=configs,
        print_passes=print_passes,
        target_opset=target_opset,
        recursive=recursive,
        specify_node_names=specify_node_names,
    )
    return graph.model


__all__ = ["convert", "convert_graph", "PassManager", "OnnxGraph"]

# make NodeProto hashable using node name
onnx.NodeProto.__hash__ = lambda self: hash(self.name)  # type: ignore

ONNXIFIER_IR_VERSION = onnx.IR_VERSION_2024_3_25
"""Currently used IR version, since most runtime supports up to this version."""

ONNXIFIER_OPSET = make_operatorsetid("", 20)
"""Currently used opset version, since most runtime supports up to this version."""
