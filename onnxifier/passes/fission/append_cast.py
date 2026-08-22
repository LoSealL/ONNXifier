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

from copy import deepcopy

from onnx import TensorProto
from onnx.helper import make_node, make_value_info

from ... import OnnxGraph
from .. import PASSES


@PASSES.register()
def append_cast(graph: OnnxGraph, to: int | str | None = None) -> OnnxGraph:
    """Append a Cast op to each graph output.

    Args:
        graph (OnnxGraph): the graph to rewrite
        to (int | str, optional): target dtype, either a ``TensorProto`` dtype
            or a dtype name like ``"FLOAT16"``. Defaults to keeping each
            output dtype unchanged (dummy cast).

    Returns:
        OnnxGraph: the rewritten graph
    """
    if isinstance(to, str):
        to = TensorProto.DataType.Value(to.upper())

    # append cast
    cast_nodes = []
    for output in list(graph.output):
        new_out_name = f"{output.name}/cast_output"
        out_type = output.type
        if to is not None:
            out_type = deepcopy(output.type)
            out_type.tensor_type.elem_type = to
        cast = make_node(
            "Cast",
            inputs=[output.name],
            outputs=[new_out_name],
            name=f"{output.name}/cast",
            to=to if to is not None else output.type.tensor_type.elem_type,
        )
        cast_nodes.append(cast)

        # update output
        graph.outputs[new_out_name] = len(graph.output)
        graph.output.append(
            make_value_info(
                name=new_out_name, type_proto=out_type, doc_string=output.doc_string
            )
        )
        graph.remove_output(output.name)

    for node in cast_nodes:
        graph.add_onnx_node(node)

    return graph
