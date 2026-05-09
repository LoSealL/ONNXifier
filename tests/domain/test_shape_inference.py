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

import threading

import onnx
import onnxscript
import pytest
from onnx.helper import make_function, make_graph, make_model, make_node, make_opsetid
from onnx.helper import make_tensor_value_info as tv
from onnxscript.onnx_opset import opset19 as op
from onnxscript.values import OnnxFunction

from onnxifier.domain.shape_inference import (
    ShapeInferenceRegistry,
    TemporaryFunctionGenerator,
    get_shape_inference,
    register_shape_inference,
)
from onnxifier.graph import OnnxGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockOpset:
    def __init__(self, domain):
        self.domain = domain


class _MockOnnxFunction:
    def __init__(self, name, domain):
        self.name = name
        self.opset = _MockOpset(domain)

    def to_function_proto(self):
        return make_function(
            domain=self.opset.domain,
            fname=self.name,
            inputs=["input_0"],
            outputs=["output_0"],
            nodes=[make_node("Identity", ["input_0"], ["output_0"])],
            opset_imports=[make_opsetid("", 13)],
        )


OnnxFunction.register(_MockOnnxFunction)


def _make_onnxscript_func(name: str = "TestOp", domain: str = "test"):
    """Return a mock OnnxFunction that returns a fresh FunctionProto each call."""
    return _MockOnnxFunction(name, domain)


def _make_model_with_custom_op(op_type: str = "TestOp", domain: str = "test"):
    """Build a minimal model containing one custom-domain node."""
    x = tv("x", onnx.TensorProto.FLOAT, [2, 64, 128])
    y = tv("y", onnx.TensorProto.FLOAT, None)
    node = make_node(op_type, ["x"], ["y"], domain=domain)
    graph = make_graph([node], "test", [x], [y])
    model = make_model(graph)
    model.opset_import.append(make_opsetid(domain, 1))
    return model


# ---------------------------------------------------------------------------
# ShapeInferenceRegistry
# ---------------------------------------------------------------------------


class TestShapeInferenceRegistry:
    def test_register_and_get(self):
        """Register an ONNXScript function and retrieve its FunctionProto."""
        registry = ShapeInferenceRegistry()
        func = _make_onnxscript_func("MambaPlugin", "trt")
        registry.register("trt", "MambaPlugin", func)

        proto = registry.get("trt", "MambaPlugin")
        assert proto is not None
        assert proto.name == "MambaPlugin"
        assert proto.domain == "trt"

    def test_get_unregistered_returns_none(self):
        """Unregistered op returns None."""
        registry = ShapeInferenceRegistry()
        assert registry.get("trt", "UnknownOp") is None

    def test_register_empty_domain_raises(self):
        """Empty domain raises ValueError."""
        registry = ShapeInferenceRegistry()
        func = _make_onnxscript_func()
        with pytest.raises(ValueError, match="domain must be non-empty"):
            registry.register("", "Op", func)

    def test_register_empty_op_type_raises(self):
        """Empty op_type raises ValueError."""
        registry = ShapeInferenceRegistry()
        func = _make_onnxscript_func()
        with pytest.raises(ValueError, match="op_type must be non-empty"):
            registry.register("trt", "", func)

    def test_register_non_onnxscript_raises(self):
        """Plain callable (not OnnxFunction) raises TypeError."""
        registry = ShapeInferenceRegistry()
        with pytest.raises(
            TypeError, match="func must be wrapped from onnxscript.script"
        ):
            registry.register("trt", "Op", lambda x: x)

    def test_get_returns_correct_function_proto(self):
        """Registry.get returns FunctionProto with overridden domain/name."""
        registry = ShapeInferenceRegistry()
        func = _make_onnxscript_func("RealName", "real.domain")
        registry.register("trt", "TestOp", func)

        proto = registry.get("trt", "TestOp")
        # Must be overridden to the registered domain/op_type,
        # not the function's original name
        assert proto.name == "TestOp"
        assert proto.domain == "trt"

    def test_list_domains(self):
        """list_domains returns all registered domains."""
        registry = ShapeInferenceRegistry()
        registry.register("trt", "Op1", _make_onnxscript_func("Op1", "trt"))
        registry.register(
            "com.microsoft",
            "Op2",
            _make_onnxscript_func("Op2", "com.microsoft"),
        )

        domains = registry.list_domains()
        assert "trt" in domains
        assert "com.microsoft" in domains

    def test_list_ops(self):
        """list_ops returns all registered ops for a domain."""
        registry = ShapeInferenceRegistry()
        registry.register("trt", "Op1", _make_onnxscript_func("Op1", "trt"))
        registry.register("trt", "Op2", _make_onnxscript_func("Op2", "trt"))

        ops = registry.list_ops("trt")
        assert "Op1" in ops
        assert "Op2" in ops

    def test_thread_safety(self):
        """Concurrent registration does not corrupt the registry."""
        registry = ShapeInferenceRegistry()
        errors = []

        def register_many():
            try:
                for i in range(100):
                    registry.register(
                        "trt",
                        f"Op{i}",
                        _make_onnxscript_func(f"Op{i}", "trt"),
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(registry.list_ops("trt")) == 100


# ---------------------------------------------------------------------------
# TemporaryFunctionGenerator
# ---------------------------------------------------------------------------


class TestTemporaryFunctionGenerator:
    def test_make_unique_name(self):
        """Generated names are unique and contain the prefix."""
        registry = ShapeInferenceRegistry()
        generator = TemporaryFunctionGenerator(registry)

        name1 = generator._make_unique_name("trt", "TestOp")
        name2 = generator._make_unique_name("com.microsoft", "TestOp")
        name3 = generator._make_unique_name("trt", "OtherOp")

        assert name1 != name2
        assert name1 != name3
        assert name2 != name3
        assert name1.startswith("_onnxifier_shape_infer")

    def test_generate_for_model_no_domain_ops(self):
        """Standard ONNX ops produce no temporary functions."""
        registry = ShapeInferenceRegistry()
        generator = TemporaryFunctionGenerator(registry)

        x = tv("x", onnx.TensorProto.FLOAT, [1, 2, 3])
        y = tv("y", onnx.TensorProto.FLOAT, [1, 2, 3])
        node = make_node("Relu", ["x"], ["y"])
        graph = make_graph([node], "test", [x], [y])
        model = make_model(graph)

        temp_functions = generator.generate_for_model(model)
        assert len(temp_functions) == 0

    def test_generate_for_model_with_registered_op(self):
        """Domain ops with registered ONNXScript functions
        generate temp protos."""
        registry = ShapeInferenceRegistry()
        func = _make_onnxscript_func("TestOp", "custom")
        registry.register("custom", "TestOp", func)

        generator = TemporaryFunctionGenerator(registry)
        model = _make_model_with_custom_op("TestOp", "custom")

        temp_functions = generator.generate_for_model(model)

        assert len(temp_functions) == 1
        assert temp_functions[0].domain == "custom"
        assert temp_functions[0].name == "TestOp"

    def test_generate_for_model_skips_existing_functions(self):
        """If the model already contains a function for the op, skip generation."""
        registry = ShapeInferenceRegistry()
        func = _make_onnxscript_func("TestOp", "custom")
        registry.register("custom", "TestOp", func)

        model = _make_model_with_custom_op("TestOp", "custom")
        # Pre-populate model with a function of the same name
        existing = make_function(
            domain="custom",
            fname="TestOp",
            inputs=["x"],
            outputs=["y"],
            nodes=[make_node("Identity", ["x"], ["y"])],
            opset_imports=[make_opsetid("", 13)],
        )
        model.functions.append(existing)

        generator = TemporaryFunctionGenerator(registry)
        temp_functions = generator.generate_for_model(model)

        # Should not generate a duplicate
        assert len(temp_functions) == 0

    def test_cleanup_removes_temp_functions(self):
        """Cleanup removes temporary functions from graph."""
        registry = ShapeInferenceRegistry()
        func = _make_onnxscript_func("TestOp", "custom")
        registry.register("custom", "TestOp", func)

        generator = TemporaryFunctionGenerator(registry)
        model = _make_model_with_custom_op("TestOp", "custom")

        temp_functions = generator.generate_for_model(model)
        assert len(model.functions) == 1

        graph = OnnxGraph(model, infer_shape=False)
        generator.cleanup(graph, temp_functions)
        assert len(graph.model.functions) == 0

    def test_generate_skips_unregistered_ops(self):
        """Domain ops without a registered inference expression are silently skipped."""
        registry = ShapeInferenceRegistry()
        generator = TemporaryFunctionGenerator(registry)

        model = _make_model_with_custom_op("UnknownOp", "custom")
        temp_functions = generator.generate_for_model(model)

        assert len(temp_functions) == 0

    def test_generate_onnxscript_conversion_failure(self):
        """If to_function_proto raises, the op is skipped
        and warning is logged."""

        class _BrokenFunc:
            name = "BrokenOp"
            opset = _MockOpset("test")

            def to_function_proto(self):
                raise RuntimeError("Conversion failed")

        OnnxFunction.register(_BrokenFunc)

        registry = ShapeInferenceRegistry()
        registry.register("test", "BrokenOp", _BrokenFunc())
        generator = TemporaryFunctionGenerator(registry)

        model = _make_model_with_custom_op("BrokenOp", "test")
        temp_functions = generator.generate_for_model(model)

        assert len(temp_functions) == 0


# ---------------------------------------------------------------------------
# Public API (decorator-based)
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_register_shape_inference_decorator(self):
        """Decorator registers ONNXScript function correctly."""

        @register_shape_inference("com.test", "MyOp")
        @onnxscript.script(default_opset=op)
        def my_shape_infer(input_0):
            return op.Identity(input_0)

        proto = get_shape_inference("com.test", "MyOp")
        assert proto is not None
        assert proto.name == "MyOp"
        assert proto.domain == "com.test"

    def test_register_shape_inference_decorator_without_args(self):
        """Decorator without explicit domain/op_type falls back to function metadata."""

        @register_shape_inference()
        @onnxscript.script(default_opset=op)
        def inferred_op(input_0):
            return op.Identity(input_0)

        # Falls back to the function's __name__ / domain
        # (which are defaults from onnxscript)
        proto = get_shape_inference("this", "inferred_op")
        assert proto is not None
        assert proto.name == "inferred_op"

    def test_get_shape_inference_unregistered(self):
        """get_shape_inference returns None for unregistered ops."""
        assert get_shape_inference("com.test", "UnknownOp") is None

    def test_register_shape_inference_validation(self):
        """Empty domain or op_type raises ValueError."""
        with pytest.raises(ValueError, match="domain must be non-empty"):
            # Apply decorator to a real function so validation runs
            decorator = register_shape_inference("", "Op")
            decorator(_make_onnxscript_func())

        with pytest.raises(ValueError, match="op_type must be non-empty"):
            decorator = register_shape_inference("domain", "")
            decorator(_make_onnxscript_func())

    def test_register_shape_inference_non_onnxscript_raises(self):
        """Decorating a plain Python function raises TypeError."""
        with pytest.raises(
            TypeError, match="func must be wrapped from onnxscript.script"
        ):

            @register_shape_inference("domain", "Op")
            def plain_func(x):
                return x


# ---------------------------------------------------------------------------
# End-to-end integration via infer_shape
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_infer_shape_with_registered_domain_op(self):
        """End-to-end shape inference works for a registered domain op."""
        from onnxifier.passes.globals.infer_shape import infer_shape

        register_shape_inference("custom", "IdentityLike")(
            _make_onnxscript_func("IdentityLike", "custom")
        )

        x = tv("x", onnx.TensorProto.FLOAT, [2, 64, 128])
        y = tv("y", onnx.TensorProto.FLOAT, None)

        node = make_node(
            "IdentityLike",
            ["x"],
            ["y"],
            domain="custom",
        )
        graph = make_graph([node], "test", [x], [y])
        model = make_model(graph)
        model.opset_import.append(make_opsetid("custom", 1))

        graph_with_shapes = infer_shape(OnnxGraph(model))
        assert graph_with_shapes.tensor_shape("y") == [2, 64, 128]

    def test_no_temp_functions_leaked(self):
        """Temporary functions are cleaned up after inference."""
        from onnxifier.passes.globals.infer_shape import infer_shape

        register_shape_inference("custom", "Passthrough")(
            _make_onnxscript_func("Passthrough", "custom")
        )

        x = tv("x", onnx.TensorProto.FLOAT, [2, 64, 128])
        weight = tv("weight", onnx.TensorProto.FLOAT, [64, 1, 3])
        context = tv("context", onnx.TensorProto.INT32, [2])
        y = tv("y", onnx.TensorProto.FLOAT, None)

        node = make_node(
            "Passthrough",
            ["x", "weight", "", "context"],
            ["y"],
            domain="custom",
        )
        graph = make_graph([node], "test", [x, weight, context], [y])
        model = make_model(graph)
        model.opset_import.append(make_opsetid("custom", 1))

        graph_with_shapes = infer_shape(OnnxGraph(model))
        assert len(graph_with_shapes.model.functions) == 0

    def test_unregistered_domain_op_graceful_fallback(self):
        """Unregistered domain ops leave shapes unknown."""
        from onnxifier.passes.globals.infer_shape import infer_shape

        x = tv("x", onnx.TensorProto.FLOAT, [2, 64, 128])
        z = tv("z", onnx.TensorProto.FLOAT, None)

        node1 = make_node("UnknownOp", ["x"], ["y"], domain="unregistered")
        node2 = make_node("Relu", ["y"], ["z"])

        graph = make_graph([node1, node2], "test", [x], [z])
        model = make_model(graph)
        model.opset_import.append(make_opsetid("unregistered", 1))

        graph_with_shapes = infer_shape(OnnxGraph(model))

        # Unregistered op's output shape should remain unknown
        y_shape, _ = graph_with_shapes.tensor_info("y")
        assert y_shape is None
        # Standard op after unknown shape remains unknown
        z_shape, _ = graph_with_shapes.tensor_info("z")
        assert z_shape is None

    def test_mixed_registered_unregistered_domain_ops(self):
        """Registered ops get shapes; unregistered ones remain unknown."""
        from onnxifier.passes.globals.infer_shape import infer_shape

        register_shape_inference("custom", "KnownOp")(
            _make_onnxscript_func("KnownOp", "custom")
        )

        x = tv("x", onnx.TensorProto.FLOAT, [2, 64, 128])
        tv("y", onnx.TensorProto.FLOAT, None)
        z = tv("z", onnx.TensorProto.FLOAT, None)

        node1 = make_node("KnownOp", ["x"], ["y"], domain="custom")
        node2 = make_node("UnknownOp", ["y"], ["z"], domain="unregistered")

        graph = make_graph([node1, node2], "test", [x], [z])
        model = make_model(graph)
        model.opset_import.append(make_opsetid("custom", 1))
        model.opset_import.append(make_opsetid("unregistered", 1))

        graph_with_shapes = infer_shape(OnnxGraph(model))

        # y is intermediate, use tensor_info to check
        y_shape, _ = graph_with_shapes.tensor_info("y")
        assert y_shape == [2, 64, 128]
        z_shape, _ = graph_with_shapes.tensor_info("z")
        assert z_shape is None

    def test_cleanup_on_inference_failure(self):
        """Temp functions are cleaned up even when inference fails."""
        from onnxifier.passes.globals.infer_shape import infer_shape

        register_shape_inference("custom", "FragileOp")(
            _make_onnxscript_func("FragileOp", "custom")
        )

        x = tv("x", onnx.TensorProto.FLOAT, [2, 64, 128])
        y = tv("y", onnx.TensorProto.FLOAT, None)

        # Create a malformed node with mismatched inputs to trigger failure
        node = make_node("FragileOp", ["x", "missing"], ["y"], domain="custom")
        graph = make_graph([node], "test", [x], [y])
        model = make_model(graph)
        model.opset_import.append(make_opsetid("custom", 1))

        # Should not raise; cleanup happens in finally block
        graph_with_shapes = infer_shape(OnnxGraph(model))
        assert len(graph_with_shapes.model.functions) == 0
