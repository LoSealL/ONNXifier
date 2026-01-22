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

from ... import OnnxGraph
from .. import PASSES


@PASSES.register()
def eliminate_unused_input(graph: OnnxGraph) -> OnnxGraph:
    """Eliminate unused input from graph"""
    inputs = set()
    for node in graph:
        inputs.update(graph.nodes[node]["pb"].input)

    pop_list = []
    for inp in graph.inputs:
        if inp not in inputs:
            i = graph.inputs[inp]
            pop_list.append((inp, i))
    for inp, i in sorted(pop_list, key=lambda x: x[1], reverse=True):
        graph.input.pop(i)
        graph.inputs.pop(inp)

    # re-assign index of inputs.
    for i, node in enumerate(graph.input):
        graph.inputs[node.name] = i
    return graph
