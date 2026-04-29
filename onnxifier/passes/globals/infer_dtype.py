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

import onnx
from onnx import NodeProto
from onnx.defs import OpSchema

from ...graph import OnnxGraph
from ...logger import debug, info
from .. import PASSES
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter
from ..utils import attribute_value, is_elewise

# Cache for interactive user inputs: {(node_name, output_name): dtype}
_user_dtype_cache: dict[tuple[str, str], int] = {}


def _parse_type_str(type_str: str) -> tuple[bool, int]:
    """Parse an ONNX type string into (is_concrete, dtype).

    Returns:
        (True, dtype) for concrete types like ``tensor(float)``.
        (False, UNDEFINED) for type variables like ``T``.
    """
    type_str = type_str.strip()
    if type_str.startswith("tensor(") and type_str.endswith(")"):
        inner = type_str[len("tensor(") : -1]
        try:
            return True, onnx.TensorProto.DataType.Value(inner.upper())
        except ValueError:
            pass
    return False, onnx.TensorProto.UNDEFINED


def _type_str_to_dtype(type_str: str) -> int:
    """Convert a concrete ONNX type string to dtype, or UNDEFINED on failure."""
    is_concrete, dtype = _parse_type_str(type_str)
    return dtype if is_concrete else onnx.TensorProto.UNDEFINED


def _get_schema_safe(node: NodeProto) -> OpSchema | None:
    """Safely retrieve ONNX OpSchema for a node."""
    try:
        return onnx.defs.get_schema(node.op_type, domain=node.domain or "")
    except onnx.defs.SchemaError:
        return None


def _collect_input_type_params(
    graph: OnnxGraph, node: NodeProto, schema: OpSchema
) -> dict[str, list[int]]:
    """Collect mapping from type parameter (e.g. ``T``) to actual input dtypes."""
    mapping: dict[str, list[int]] = {}
    for i, inp_decl in enumerate(schema.inputs):
        if i >= len(node.input) or not node.input[i]:
            continue
        _, inp_dtype = graph.tensor_info(node.input[i])
        is_concrete, _ = _parse_type_str(inp_decl.type_str)
        if not is_concrete:
            param = inp_decl.type_str
            mapping.setdefault(param, []).append(inp_dtype)
    return mapping


def _get_candidates_from_schema(schema: OpSchema | None, output_idx: int) -> list[int]:
    """Get candidate dtypes from schema type constraints for a given output."""
    if schema is None:
        return []

    if (
        len(schema.outputs) == 1
        and schema.outputs[0].option == OpSchema.FormalParameterOption.Variadic
    ):
        out_decl = schema.outputs[0]
    elif output_idx < len(schema.outputs):
        out_decl = schema.outputs[output_idx]
    else:
        return []

    is_concrete, dtype = _parse_type_str(out_decl.type_str)
    if is_concrete:
        return [dtype]

    type_param = out_decl.type_str
    for tc in schema.type_constraints:
        if tc.type_param_str == type_param:
            candidates = [_type_str_to_dtype(t) for t in tc.allowed_type_strs]
            return [c for c in candidates if c != onnx.TensorProto.UNDEFINED]
    return []


def _infer_from_special_ops(graph: OnnxGraph, node: NodeProto) -> list[int] | None:
    """Infer output dtypes for ops whose schema alone is insufficient.

    Returns a list of inferred dtypes (one per output) or ``None`` if the op is
    not handled specially.
    """
    op_type = node.op_type

    if op_type == "Cast":
        to = None
        for attr in node.attribute:
            if attr.name == "to":
                to = attribute_value(attr)
                break
        if to is not None:
            assert isinstance(to, int)
            return [to]
        return None

    if op_type == "CastLike":
        if len(node.input) > 1 and node.input[1]:
            _, dtype = graph.tensor_info(node.input[1])
            if dtype != onnx.TensorProto.UNDEFINED:
                return [dtype]
        return None

    if op_type == "Constant":
        for attr in node.attribute:
            if attr.name == "value":
                return [attr.t.data_type]
            if attr.name == "value_float":
                return [onnx.TensorProto.FLOAT]
            if attr.name == "value_floats":
                return [onnx.TensorProto.FLOAT]
            if attr.name == "value_int":
                return [onnx.TensorProto.INT64]
            if attr.name == "value_ints":
                return [onnx.TensorProto.INT64]
            if attr.name == "value_string":
                return [onnx.TensorProto.STRING]
            if attr.name == "value_strings":
                return [onnx.TensorProto.STRING]
        return None

    if op_type in ("RandomUniform", "RandomNormal"):
        dtype = onnx.TensorProto.FLOAT
        for attr in node.attribute:
            if attr.name == "dtype":
                dtype = attribute_value(attr)
                assert isinstance(dtype, int)
                break
        return [dtype]

    if op_type in ("RandomUniformLike", "RandomNormalLike"):
        dtype = None
        for attr in node.attribute:
            if attr.name == "dtype":
                dtype = attribute_value(attr)
                break
        if dtype is not None:
            assert isinstance(dtype, int)
            return [dtype]
        if node.input[0]:
            _, dtype = graph.tensor_info(node.input[0])
            if dtype != onnx.TensorProto.UNDEFINED:
                return [dtype]
        return None

    if op_type == "EyeLike":
        dtype = None
        for attr in node.attribute:
            if attr.name == "dtype":
                dtype = attribute_value(attr)
                break
        if dtype is not None:
            assert isinstance(dtype, int)
            return [dtype]
        if node.input[0]:
            _, dtype = graph.tensor_info(node.input[0])
            if dtype != onnx.TensorProto.UNDEFINED:
                return [dtype]
        return None

    if op_type == "ConstantOfShape":
        for attr in node.attribute:
            if attr.name == "value":
                return [attr.t.data_type]
        return [onnx.TensorProto.FLOAT]

    if op_type == "Range":
        if node.input[0]:
            _, dtype = graph.tensor_info(node.input[0])
            if dtype != onnx.TensorProto.UNDEFINED:
                return [dtype]
        return None

    return None


def _infer_from_schema(
    graph: OnnxGraph, node: NodeProto, schema: OpSchema | None, output_idx: int
) -> int:
    """Infer dtype for a specific output using the ONNX schema."""
    if schema is None:
        return onnx.TensorProto.UNDEFINED

    # Determine the corresponding schema output declaration
    if (
        len(schema.outputs) == 1
        and schema.outputs[0].option == OpSchema.FormalParameterOption.Variadic
    ):
        out_decl = schema.outputs[0]
    elif output_idx < len(schema.outputs):
        out_decl = schema.outputs[output_idx]
    else:
        return onnx.TensorProto.UNDEFINED

    # If the schema specifies a concrete type, use it directly
    is_concrete, dtype = _parse_type_str(out_decl.type_str)
    if is_concrete:
        return dtype

    # Type variable: try to map from inputs
    type_param = out_decl.type_str
    input_mapping = _collect_input_type_params(graph, node, schema)

    if type_param in input_mapping:
        dtypes = set(input_mapping[type_param])
        dtypes.discard(onnx.TensorProto.UNDEFINED)
        if len(dtypes) == 1:
            return dtypes.pop()
        if len(dtypes) > 1:
            # Ambiguous: pick the first concrete dtype encountered
            return next(iter(dtypes))

    return onnx.TensorProto.UNDEFINED


def _guess_output_dtype(
    graph: OnnxGraph,
    node: NodeProto,
    schema: OpSchema | None,
    output_idx: int,
) -> int:
    """Heuristic fallback when schema inference fails."""
    # Special ops first
    special = _infer_from_special_ops(graph, node)
    if special is not None:
        if (
            schema is not None
            and len(schema.outputs) == 1
            and schema.outputs[0].option == OpSchema.FormalParameterOption.Variadic
        ):
            return special[0] if special else onnx.TensorProto.UNDEFINED
        if output_idx < len(special):
            return special[output_idx]
        return onnx.TensorProto.UNDEFINED

    # Element-wise ops usually preserve the first input dtype
    if is_elewise(node):
        input_dtypes = []
        for i in node.input:
            if i:
                _, dt = graph.tensor_info(i)
                if dt != onnx.TensorProto.UNDEFINED:
                    input_dtypes.append(dt)
        if input_dtypes:
            if len(set(input_dtypes)) == 1:
                return input_dtypes[0]
            return input_dtypes[0]

    # Single-input ops usually preserve dtype
    non_empty_inputs = [i for i in node.input if i]
    if len(non_empty_inputs) == 1:
        _, dt = graph.tensor_info(non_empty_inputs[0])
        if dt != onnx.TensorProto.UNDEFINED:
            return dt

    # Fallback: use the first non-UNDEFINED input dtype
    for i in node.input:
        if i:
            _, dt = graph.tensor_info(i)
            if dt != onnx.TensorProto.UNDEFINED:
                return dt

    return onnx.TensorProto.UNDEFINED


def _ask_user_for_dtype(
    node_name: str, op_type: str, output_name: str, candidates: list[int]
) -> int:
    """Prompt the user to select a dtype when inference fails."""
    cache_key = (node_name, output_name)
    if cache_key in _user_dtype_cache:
        return _user_dtype_cache[cache_key]

    print("\n[InferDType] Cannot infer dtype for:")
    print(f"  Node : {node_name} ({op_type})")
    print(f"  Output: {output_name}")
    if candidates:
        print("  Candidate types:")
        for idx, dt in enumerate(candidates):
            dt_name = onnx.TensorProto.DataType.Name(dt)
            print(f"    {idx}: {dt_name} ({dt})")
    else:
        print("  No schema candidates. Common: FLOAT(1), INT64(7), INT32(6)")

    while True:
        try:
            choice = input(
                "Please enter the dtype index or name "
                "(e.g. 'FLOAT', 'INT64'), or press Enter to skip: "
            ).strip()
            if not choice:
                return onnx.TensorProto.UNDEFINED

            # Try index first
            try:
                idx = int(choice)
                if 0 <= idx < len(candidates):
                    _user_dtype_cache[cache_key] = candidates[idx]
                    return candidates[idx]
            except ValueError:
                pass

            # Try dtype name
            dt = onnx.TensorProto.DataType.Value(choice.upper())
            _user_dtype_cache[cache_key] = dt
            return dt

        except (EOFError, KeyboardInterrupt):
            return onnx.TensorProto.UNDEFINED
        except ValueError:
            print("Invalid choice, please try again.")


@PASSES.register("infer_dtype", deps=["infer_shape"])
class InferDTypeRewriter(Rewriter):
    """Verify and repair node output dtypes using ONNX schema and heuristics.

    This pass depends on ``infer_shape`` because it reuses existing shape info
    and only updates the dtype field of ``value_info``.

    Args:
        interactive: If ``True``, prompt the user for a dtype when automatic
            inference fails. Defaults to ``False``.
    """

    def __init__(self):
        super().__init__(pattern=SingleNodePattern())

    def rewrite(
        self,
        graph: OnnxGraph,
        nodes: list[NodeProto],
        *_a,
        interactive: bool = False,
        **_kw,
    ):
        node = nodes[0]
        schema = _get_schema_safe(node)
        if schema is not None:
            debug(
                "Found node schema: op=%s schema=%s domain=%s since_version=%s",
                node.op_type,
                schema.name,
                node.domain,
                schema.since_version,
            )

        for output_idx, output_name in enumerate(node.output):
            if not output_name:
                continue

            shape, current_dtype = graph.tensor_info(output_name)

            # Try schema-based inference
            inferred_dtype = _infer_from_schema(graph, node, schema, output_idx)

            # Fallback to heuristics / special ops
            if inferred_dtype == onnx.TensorProto.UNDEFINED:
                inferred_dtype = _guess_output_dtype(graph, node, schema, output_idx)

            # Interactive fallback
            if inferred_dtype == onnx.TensorProto.UNDEFINED and interactive:
                candidates = _get_candidates_from_schema(schema, output_idx)
                inferred_dtype = _ask_user_for_dtype(
                    node.name, node.op_type, output_name, candidates
                )

            if inferred_dtype == onnx.TensorProto.UNDEFINED:
                continue

            if current_dtype == onnx.TensorProto.UNDEFINED:
                debug(
                    "[InferDType] Set %s of %s(%s) to %s",
                    output_name,
                    node.name,
                    node.op_type,
                    onnx.TensorProto.DataType.Name(inferred_dtype),
                )
                graph.set_value_info(
                    output_name, shape if shape is not None else (), inferred_dtype
                )
            elif current_dtype != inferred_dtype:
                info(
                    "[InferDType] Dtype mismatch for %s of %s(%s): "
                    "current=%s, inferred=%s. Fixing...",
                    output_name,
                    node.name,
                    node.op_type,
                    onnx.TensorProto.DataType.Name(current_dtype),
                    onnx.TensorProto.DataType.Name(inferred_dtype),
                )
                graph.set_value_info(
                    output_name, shape if shape is not None else (), inferred_dtype
                )
