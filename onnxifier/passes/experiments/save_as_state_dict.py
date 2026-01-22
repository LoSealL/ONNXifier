"""
Copyright (C) 2025 The ONNXIFIER Authors.

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

# pylint: disable=arguments-differ

from pathlib import Path
from typing import List, Sequence

import numpy as np
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph, logger
from ...utils import legalize_name
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


@PASSES.register("save_as_state_dict")
class SaveAsStateDictRewriter(Rewriter):
    """
    Save model as state dict.

    This pass will save the model as a state dict, which is a dictionary that
    maps each parameter name to its value.
    The state dict will be saved in the specified directory.
    """

    def __init__(
        self,
        interested_nodes: Sequence[str] = ("Conv", "ConvTranspose", "MatMul", "Gemm"),
        save_dir: str = ".",
    ):
        patterns = [SingleNodePattern(i) for i in interested_nodes]
        pattern = patterns[0]
        for p in patterns[1:]:
            pattern |= p
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        self.states: dict[str, np.ndarray] = {}
        super().__init__(pattern=pattern)
        self.register_post_hook(self.save_states)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto]):
        node = nodes[0]
        name = node.name.strip("/").replace("/", ".")
        if node.op_type in ("Conv", "ConvTranspose"):
            has_bias = len(node.input) > 2 and node.input[2] != ""
            weight = self.get_value_or_die(node.input[1])
            self.states[name + ".weight"] = weight
            if has_bias:
                bias = self.get_value_or_die(node.input[2])
                self.states[name + ".bias"] = bias
        elif node.op_type in ("MatMul", "Gemm"):
            a = self.get_value(node.input[0])
            b = self.get_value(node.input[1])
            if a is None and b is None:
                return
            elif a is not None and b is None:
                self.states[name + ".weight"] = a
            elif b is not None and a is None:
                self.states[name + ".weight"] = b
            elif a is not None and b is not None:
                logger.warning(
                    f"Both inputs to {node.op_type} node '{name}' are constant. "
                    "Saving the second input as '.weight'."
                )
                self.states[name + ".weight"] = b

    def save_states(self, graph: OnnxGraph) -> OnnxGraph:
        try:
            import torch
        except ImportError:
            logger.warning("Please install torch to save model as state dict.")
            return graph

        states = {k: torch.from_numpy(v) for k, v in self.states.items()}
        model_name = legalize_name(graph.name)
        torch.save(states, Path(self.save_dir) / f"{model_name}.pth")
        return graph
