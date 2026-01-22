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

from onnx.helper import make_node, make_value_info

from ... import OnnxGraph
from .. import PASSES


@PASSES.register()
def append_cast(graph: OnnxGraph) -> OnnxGraph:
    """Append identity op to the output.

    Args:
        graph (OnnxGraph): the graph to rewrite

    Returns:
        OnnxGraph: the rewritten graph
    """

    # append cast
    cast_nodes = []
    for output in list(graph.output):
        new_out_name = f"{output.name}/cast_output"
        cast = make_node(
            "Cast",
            inputs=[output.name],
            outputs=[new_out_name],
            name=f"{output.name}/cast",
            to=output.type.tensor_type.elem_type,  # dummy cast
        )
        cast_nodes.append(cast)

        # update output
        graph.outputs[new_out_name] = len(graph.output)
        graph.output.append(
            make_value_info(
                name=new_out_name, type_proto=output.type, doc_string=output.doc_string
            )
        )
        graph.remove_output(output.name)

    for node in cast_nodes:
        graph.add_onnx_node(node)

    return graph
