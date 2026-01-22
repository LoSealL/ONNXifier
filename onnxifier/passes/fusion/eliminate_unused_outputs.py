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

from ... import OnnxGraph, logger
from .. import PASSES


@PASSES.register()
def eliminate_unused_outputs(graph: OnnxGraph) -> OnnxGraph:
    """Eliminate unused input from graph"""
    for k in list(graph.outputs):
        # pylint: disable=protected-access
        if k not in graph._out_to_node:
            logger.debug(f"Removing unused output: {k}")
            graph.remove_output(k)
    return graph
