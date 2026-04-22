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

import re
from collections.abc import Sequence

import onnx
from onnx import NodeProto
from onnx.helper import make_node, make_tensor_type_proto, make_value_info

from ...graph import OnnxGraph
from ...logger import warning
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("debug_node")
class DebugNodeRewriter(Rewriter):
    """Expose matched node inputs and outputs as model outputs for debugging.

    Notes:
        - `node_types` supports comma-separated regular expressions.
        - Inputs that are constants, foldable constant paths, or graph inputs are
          skipped.
        - Existing graph outputs are not duplicated.
        - Outputs are exported even without shape info; only dtype is required.
        - Added outputs are aliased through Identity nodes and prefixed for clarity.
    """

    def __init__(self):
        super().__init__(SingleNodePattern())
        self._compiled_patterns: dict[tuple[str, ...], list[re.Pattern[str]]] = {}
        self._reserved_tensor_names: set[str] = set()

    def rewrite(
        self,
        graph: OnnxGraph,
        nodes: list[NodeProto],
        *_v,
        node_types: str | Sequence[str] = "",
        prefix: str = "debug/",
        **_kw,
    ):
        node = nodes[0]
        if not self._match_op_type(node, node_types):
            return

        self._init_reserved_tensor_names(graph)
        for index, input_name in enumerate(node.input):
            if not input_name or self._is_constant_path(input_name):
                continue
            self._export_tensor(
                graph,
                node,
                tensor_name=input_name,
                prefix=prefix,
                tag=f"input{index}",
            )
        for index, output_name in enumerate(node.output):
            if not output_name or output_name in graph.outputs:
                continue
            self._export_tensor(
                graph,
                node,
                tensor_name=output_name,
                prefix=prefix,
                tag=f"output{index}",
            )

    def _init_reserved_tensor_names(self, graph: OnnxGraph):
        if self._reserved_tensor_names:
            return
        self._reserved_tensor_names.update(graph.inputs)
        self._reserved_tensor_names.update(graph.outputs)
        self._reserved_tensor_names.update(graph.initializers)
        # pylint: disable=protected-access
        self._reserved_tensor_names.update(graph._out_to_node)

    def _match_op_type(self, node: NodeProto, node_types: str | Sequence[str]) -> bool:
        patterns = self._compile_patterns(node_types)
        return bool(patterns) and any(
            pattern.search(node.op_type) for pattern in patterns
        )

    def _compile_patterns(
        self, node_types: str | Sequence[str]
    ) -> list[re.Pattern[str]]:
        raw_patterns: list[str] = []
        if isinstance(node_types, str):
            raw_patterns.extend(node_types.split(","))
        else:
            for pattern in node_types:
                raw_patterns.extend(str(pattern).split(","))
        cache_key = tuple(
            pattern.strip() for pattern in raw_patterns if pattern.strip()
        )
        if cache_key not in self._compiled_patterns:
            self._compiled_patterns[cache_key] = [re.compile(p) for p in cache_key]
        return self._compiled_patterns[cache_key]

    def _is_constant_path(self, tensor_name: str) -> bool:
        try:
            return self.get_value(tensor_name) is not None
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            return False

    def _export_tensor(
        self,
        graph: OnnxGraph,
        node: NodeProto,
        tensor_name: str,
        prefix: str,
        tag: str,
    ):
        # Skip if already a graph output, graph input, or constant
        if tensor_name in graph.outputs or tensor_name in graph.inputs:
            return
        debug_output_name = f"{prefix}{node.name}/{tag}"
        if debug_output_name in self._reserved_tensor_names:
            return

        shape, dtype = graph.tensor_info(tensor_name)
        if dtype == onnx.TensorProto.UNDEFINED:
            warning(
                "Skip debug output %s because dtype is unknown.",
                tensor_name,
            )
            return

        identity = make_node(
            "Identity",
            [tensor_name],
            [debug_output_name],
            name=f"{node.name}/debug/{tag}",
        )
        self += identity

        graph.outputs[debug_output_name] = len(graph.output)
        graph.output.append(
            make_value_info(
                debug_output_name,
                make_tensor_type_proto(dtype, shape if shape else []),
            )
        )
        self._reserved_tensor_names.add(debug_output_name)
