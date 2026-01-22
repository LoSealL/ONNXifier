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

from contextlib import suppress

from ... import OnnxGraph
from .. import L3

with suppress(ImportError):
    import onnxsim

    @L3.register(
        name="onnxsim",
        patch=[
            "canonicalize_conv_autopad",
            "eliminate_unused_input",
            "eliminate_unused_outputs",
        ],
    )
    def onnx_simplifier(graph: OnnxGraph):
        """Simplify onnx graph"""
        model_sim, succeed = onnxsim.simplify(graph.model, skip_shape_inference=True)
        if succeed:
            # FIXME: onnxsim ignores functions
            model_sim.functions.extend(graph.functions.values())
            return OnnxGraph(model_sim, base_dir=graph.external_base)
        return graph
