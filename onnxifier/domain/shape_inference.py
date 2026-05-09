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

import hashlib
import threading
from typing import Any

import onnx
from onnxscript.values import OnnxFunction

from ..graph import OnnxGraph
from ..logger import warning


class ShapeInferenceRegistry:
    """Registry for custom domain shape inference expressions.

    Maps (domain, op_type) pairs to shape inference functions.
    Thread-safe for concurrent registration.
    """

    def __init__(self) -> None:
        self._registry: dict[tuple[str, str], OnnxFunction[..., Any]] = {}
        self._lock = threading.RLock()

    def register(self, domain: str, op_type: str, func: OnnxFunction[..., Any]) -> None:
        """Register a shape inference function for a custom domain op.

        Args:
            domain: Domain name (e.g., "trt", "com.microsoft").
            op_type: Op type name (e.g., "MambaPlugin").
            func: Shape inference function (ONNXScript or Python callable).

        Raises:
            ValueError: If domain or op_type is empty.
            TypeError: If func is not callable.
        """
        if not domain:
            raise ValueError("domain must be non-empty")
        if not op_type:
            raise ValueError("op_type must be non-empty")
        if not isinstance(func, OnnxFunction):
            raise TypeError("func must be wrapped from onnxscript.script")

        with self._lock:
            self._registry[(domain, op_type)] = func

    def _get_raw(self, domain: str, op_type: str) -> OnnxFunction[..., Any] | None:
        """Retrieve the raw ONNXScript function (for internal use)."""
        with self._lock:
            return self._registry.get((domain, op_type))

    def get(self, domain: str, op_type: str) -> onnx.FunctionProto | None:
        """Retrieve a registered shape inference function.

        Args:
            domain: Domain name.
            op_type: Op type name.

        Returns:
            Registered function or None if not found.
        """
        with self._lock:
            func = self._registry.get((domain, op_type))
            if func is None:
                return
            onnxfunc = func.to_function_proto()
            if onnxfunc.name != op_type:
                onnxfunc.name = op_type
            if onnxfunc.domain != domain:
                onnxfunc.domain = domain
            return onnxfunc

    def list_domains(self) -> list[str]:
        """Return all registered domains.

        Returns:
            List of domain names.
        """
        with self._lock:
            return list({domain for domain, _ in self._registry})

    def list_ops(self, domain: str) -> list[str]:
        """Return all registered ops for a domain.

        Args:
            domain: Domain name.

        Returns:
            List of op type names.
        """
        with self._lock:
            return [op_type for d, op_type in self._registry if d == domain]


class TemporaryFunctionGenerator:
    """Generates temporary ONNX FunctionProtos for domain ops.

    Creates temporary functions that ONNX's native shape inference can use
    to infer output shapes for custom domain ops. Functions are generated
    from registered shape inference expressions and cleaned up after use.
    """

    def __init__(self, registry: ShapeInferenceRegistry | None = None) -> None:
        self._registry = registry or _GLOBAL_REGISTRY
        self._temp_prefix = "_onnxifier_shape_infer"
        # Schemas that were deregistered to allow temp function inference
        self._deregistered_schemas: list[tuple[onnx.defs.OpSchema, str, int, str]] = []

    def _make_unique_name(self, domain: str, op_type: str) -> str:
        """Generate a unique function name for temporary shape inference functions.

        Args:
            domain: Domain name.
            op_type: Op type name.

        Returns:
            Unique function name.
        """
        key = f"{domain}:{op_type}"
        hash_val = hashlib.md5(key.encode()).hexdigest()[:8]
        return f"{self._temp_prefix}_{domain.replace('.', '_')}_{op_type}_{hash_val}"

    def generate_for_model(self, model: onnx.ModelProto) -> list[onnx.FunctionProto]:
        """Scan model for domain ops and generate temporary FunctionProtos.

        Temporarily deregisters any existing ONNX schemas for these ops so that
        ONNX's native shape inference uses our temporary functions instead.

        Args:
            model: ONNX model.

        Returns:
            List of generated temporary functions.
        """
        if not model.graph:
            return []

        temp_functions: list[onnx.FunctionProto] = []
        registered_ops: set[tuple[str, str]] = set()
        existing_funcs: set[tuple[str, str]] = set()

        # Check for existing functions with same (domain, op_type)
        for func in model.functions:
            existing_funcs.add((func.domain, func.name))

        # First pass: identify which ops need temp functions
        ops_needing_temps: list[tuple[str, str, onnx.NodeProto]] = []
        for node in model.graph.node:
            # Skip standard ONNX ops
            if node.domain in ("", "ai.onnx", "ai.onnx.ml"):
                continue

            key = (node.domain, node.op_type)
            if key in registered_ops:
                continue

            func = self._registry._get_raw(*key)
            if func is None:
                continue

            # Skip if model already has a function for this op
            if key in existing_funcs:
                continue

            registered_ops.add(key)
            ops_needing_temps.append((node.domain, node.op_type, node))

        if not ops_needing_temps:
            return []

        # Deregister existing schemas so ONNX uses temp functions
        for domain, op_type, _ in ops_needing_temps:
            if onnx.defs.has(op_type, domain):
                try:
                    schema = onnx.defs.get_schema(op_type, domain)
                    onnx.defs.deregister_schema(op_type, schema.since_version, domain)
                    self._deregistered_schemas.append(
                        (schema, op_type, schema.since_version, domain)
                    )
                except Exception:
                    pass  # Schema might not be deregisterable, ignore

        # Generate temporary functions
        for domain, op_type, node in ops_needing_temps:
            func = self._registry._get_raw(domain, op_type)
            if func is None:
                continue

            func_proto = self._generate_from_callable(domain, op_type, func)
            if func_proto is not None:
                temp_functions.append(func_proto)
                model.functions.append(func_proto)

        return temp_functions

    def _generate_from_callable(
        self,
        domain: str,
        op_type: str,
        func: OnnxFunction[..., Any],
    ) -> onnx.FunctionProto | None:
        """Generate a FunctionProto from a Python callable.

        Creates a simple Identity-based function where each output is
        mapped to a specific input. The shape source mapping determines
        which input's shape is used for each output.

        Args:
            domain: Domain name.
            op_type: Op type name.
            func: Shape inference callable (used for ONNXScript detection).

        Returns:
            FunctionProto or None if generation fails.
        """

        try:
            func_proto = func.to_function_proto()
            # Override domain and name to match the op being replaced
            func_proto.domain = domain
            func_proto.name = op_type
            return func_proto
        except Exception as e:
            warning(f"ONNXScript function conversion failed: {e}. ")
        return None

    def cleanup(
        self, graph: OnnxGraph, temp_functions: list[onnx.FunctionProto]
    ) -> None:
        """Remove temporary functions from the model and re-register schemas.

        Args:
            graph: onnx graph.
            temp_functions: List of temporary functions to remove.
        """
        # Re-register any schemas that were deregistered
        for schema, op_type, version, domain in self._deregistered_schemas:
            try:
                onnx.defs.register_schema(schema)
            except Exception:
                pass  # Schema might already be registered, ignore
        self._deregistered_schemas.clear()

        if not temp_functions:
            return

        for f in temp_functions:
            graph.functions.pop(f.name, None)


# Global registry instance
_GLOBAL_REGISTRY = ShapeInferenceRegistry()


def register_shape_inference(
    domain: str | None = None,
    op_type: str | None = None,
):
    """Register a shape inference function for a custom domain op.

    Args:
        domain: Domain name (e.g., "trt", "com.microsoft").
        op_type: Op type name (e.g., "CausalConv1d").

    Raises:
        ValueError: If domain or op_type is empty, or if shape_source keys/values
            are not non-negative integers.
        TypeError: If func is not callable.
    """

    def _wrapper(func: OnnxFunction[..., Any]):
        func_domain = domain if domain is not None else func.opset.domain
        func_name = op_type if op_type is not None else func.name
        _GLOBAL_REGISTRY.register(func_domain, func_name, func)
        return func

    return _wrapper


def get_shape_inference(domain: str, op_type: str):
    """Retrieve a registered shape inference function.

    Args:
        domain: Domain name.
        op_type: Op type name.

    Returns:
        Registered function or None if not found.
    """
    return _GLOBAL_REGISTRY.get(domain, op_type)
