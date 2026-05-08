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

from abc import ABCMeta
from collections.abc import Sequence

import onnx
from onnx import NodeProto, TensorProto
from onnx.helper import make_tensor_type_proto, make_value_info

from .... import OnnxGraph
from ....domain.trt import TRT_IR_DOMAIN
from ...rewriter import Rewriter


def elem_type_from_schema(schema: onnx.defs.OpSchema, input_name: str) -> int:
    """Infer ONNX tensor element type from an OpSchema input declaration."""
    constraint = {
        i.type_param_str: i.allowed_type_strs for i in schema.type_constraints
    }
    for inp in schema.inputs:
        if inp.name != input_name:
            continue
        type_str = inp.type_str
        if type_str in constraint:
            type_str = constraint[type_str][0]
        if not type_str.startswith("tensor(") or not type_str.endswith(")"):
            break
        elem_name = type_str[len("tensor(") : -1].upper()
        return TensorProto.DataType.Value(elem_name)
    raise ValueError(f"Cannot infer tensor element type for input '{input_name}'.")


class EnsureTensorRTDomain(Rewriter, metaclass=ABCMeta):
    """Provide a mixin method to append trt domain to opset_import"""

    def __init__(self, pattern):
        super().__init__(pattern)
        self.register_post_hook(self._ensure_trt_domain)

    @staticmethod
    def _ensure_trt_domain(graph: OnnxGraph):
        graph._model.opset_import.append(TRT_IR_DOMAIN)
        return graph

    def ensure_inputs_outputs(self, node: NodeProto, schema: onnx.defs.OpSchema):
        """Insert inputs and outputs to node following op schema."""
        for ind, i in enumerate(schema.inputs):
            if len(node.input) <= ind:
                node.input.append("")
            if node.input[ind] == "":
                node.input[ind] = f"{node.name}/{i.name}"
        for ind, i in enumerate(schema.outputs):
            if len(node.output) <= ind:
                node.output.append("")
            if node.output[ind] == "":
                node.output[ind] = f"{node.name}/{i.name}"

    def mark_graph_input(self, name: str, dtype: int, shape: Sequence[int | str]):
        """Mark tensor {name} as graph input with given dtype and shape."""

        graph = self.graph
        if name in graph.inputs:
            # check dtype and shape
            input_shape, input_dtype = graph.tensor_info(name)
            if input_dtype != TensorProto.UNDEFINED and input_dtype != dtype:
                raise ValueError(
                    f"Input {name} already exists with different dtype: "
                    f"{input_dtype} vs {dtype}"
                )
            if input_shape is not None and len(input_shape) != len(shape):
                raise ValueError(
                    f"Input {name} already exists with different shape rank: "
                    f"{len(input_shape)} vs {len(shape)}"
                )
            if input_shape:
                for i, (dim1, dim2) in enumerate(zip(input_shape, shape)):
                    if isinstance(dim1, int) and isinstance(dim2, int) and dim1 != dim2:
                        raise ValueError(
                            f"Input {name} already exists with different shape at dim "
                            f"{i}: {dim1} vs {dim2}"
                        )
            return
        graph.set_value_info(name, shape, dtype)
        graph.inputs[name] = len(graph.inputs)
        graph.input.append(make_value_info(name, make_tensor_type_proto(dtype, shape)))

    def mark_graph_output(self, name: str, dtype: int, shape: Sequence[int | str]):
        """Mark tensor {name} as graph output with given dtype and shape."""

        graph = self.graph
        if name in graph.outputs:
            # check dtype and shape
            output_shape, output_dtype = graph.tensor_info(name)
            if output_dtype != TensorProto.UNDEFINED and output_dtype != dtype:
                raise ValueError(
                    f"Output {name} already exists with different dtype: "
                    f"{output_dtype} vs {dtype}"
                )
            if output_shape is not None and len(output_shape) != len(shape):
                raise ValueError(
                    f"Output {name} already exists with different shape rank: "
                    f"{len(output_shape)} vs {len(shape)}"
                )
            if output_shape:
                for i, (dim1, dim2) in enumerate(zip(output_shape, shape)):
                    if isinstance(dim1, int) and isinstance(dim2, int) and dim1 != dim2:
                        raise ValueError(
                            f"Output {name} already exists with different shape at dim "
                            f"{i}: {dim1} vs {dim2}"
                        )
            return
        graph.set_value_info(name, shape, dtype)
        graph.outputs[name] = len(graph.outputs)
        graph.output.append(make_value_info(name, make_tensor_type_proto(dtype, shape)))
