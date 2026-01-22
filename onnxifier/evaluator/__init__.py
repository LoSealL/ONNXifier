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

Choose a proper backend to evaluate ONNX models.
1. ReferenceEvaluator: no extra dependency, but extremely slow.
2. OpenVINO runtime: best for Intel platforms.
3. OnnxRuntime: best compatibility and balanced performance.
"""

# pylint: disable=import-outside-toplevel

import os
import warnings
from contextlib import suppress
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple

import numpy as np
import onnx
from onnx.helper import tensor_dtype_to_np_dtype


def _get_eval_onnx(model):
    from onnx.reference import ReferenceEvaluator

    if not isinstance(model, onnx.ModelProto):
        model = onnx.load_model(model)

    inputs = {}
    outputs = {}
    for i in model.graph.input:
        shape = [int(j.dim_value) for j in i.type.tensor_type.shape.dim]
        inputs[i.name] = dict(
            shape=shape,
            dtype=tensor_dtype_to_np_dtype(i.type.tensor_type.elem_type),
        )
    for i in model.graph.output:
        shape = [int(j.dim_value) for j in i.type.tensor_type.shape.dim]
        outputs[i.name] = dict(
            shape=shape,
            dtype=tensor_dtype_to_np_dtype(i.type.tensor_type.elem_type),
        )

    model = ReferenceEvaluator(model)

    def _run_code(output_names, inputs_feed):
        results = model.run(output_names, inputs_feed)
        return [np.array(i) for i in results]

    return _run_code, inputs, outputs


def _get_eval_onnxruntime(model):
    import onnxruntime

    if isinstance(model, onnx.ModelProto):
        sess = onnxruntime.InferenceSession(model.SerializeToString())
    else:
        sess = onnxruntime.InferenceSession(model)

    def _run_code(
        output_names: Sequence[str] | None, inputs_feed: Dict[str, np.ndarray]
    ) -> List[Any]:
        return list(sess.run(output_names, inputs_feed))

    def _data_mapping():
        mapping = {}
        for dtype in (
            np.float64,
            np.float32,
            np.half,
            np.uint64,
            np.uint32,
            np.uint16,
            np.uint8,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.bool_,
        ):
            mapping[
                onnxruntime.OrtValue.ortvalue_from_numpy(np.ones([], dtype)).data_type()
            ] = dtype
        return mapping

    inputs = {}
    outputs = {}
    mapping = _data_mapping()
    for i in sess.get_inputs():
        inputs[i.name] = dict(shape=tuple(i.shape), dtype=mapping[i.type])
    for i in sess.get_outputs():
        outputs[i.name] = dict(shape=tuple(i.shape), dtype=mapping[i.type])

    return _run_code, inputs, outputs


def _get_eval_openvino(model):
    import openvino

    core = openvino.Core()
    with TemporaryDirectory() as tmpdir:
        if isinstance(model, onnx.ModelProto):
            onnx.save_model(model, f"{tmpdir}/model.onnx")
            network = core.read_model(f"{tmpdir}/model.onnx")
            model = openvino.compile_model(network)
        else:
            network = core.read_model(model)
            model = core.compile_model(network)

    def _run_code(output_names, inputs_feed):
        input_names = {i.name for i in network.get_parameters()}
        if missing_names := input_names.difference(inputs_feed.keys()):
            warnings.warn(f"Missing inputs: {missing_names}")
        outputs = model(inputs_feed)
        if output_names:
            return [outputs[name] for name in output_names]
        return [outputs[i] for i, _ in enumerate(outputs)]

    inputs = {}
    outputs = {}
    for i, inp in enumerate(network.inputs):
        if inp.names:
            key = inp.get_any_name()
        else:
            key = i
        inputs[key] = dict(
            shape=tuple(inp.shape), dtype=inp.get_element_type().to_dtype()
        )
    for i, oup in enumerate(network.outputs):
        if oup.names:
            key = oup.get_any_name()
        else:
            key = i
        outputs[key] = dict(
            shape=tuple(oup.shape), dtype=oup.get_element_type().to_dtype()
        )

    return _run_code, inputs, outputs


# pyright: ignore[reportReturnType]
def _get_eval_backend(
    backend: str, model: str | os.PathLike | onnx.ModelProto
) -> Tuple[
    Callable[[Sequence[str] | None, Dict[str, np.ndarray]], List[np.ndarray]],
    Dict,
    Dict,
]:
    if backend == "onnx":
        return _get_eval_onnx(model)
    elif backend == "onnxruntime":
        return _get_eval_onnxruntime(model)
    elif backend == "openvino":
        return _get_eval_openvino(model)

    raise NotImplementedError(f"Unsupported backend: {backend}")


class Evaluator:
    """An evaluator wraps different backends to evaluate ONNX models."""

    def __init__(
        self,
        model: str | os.PathLike | onnx.ModelProto,
        backend: Literal["auto", "openvino", "onnxruntime", "onnx"] = "auto",
    ):
        if backend.lower() == "auto":
            with suppress(ImportError):
                # pylint: disable=unused-import
                import onnxruntime  # noqa: F401

                backend = "onnxruntime"
        if backend.lower() == "auto":
            with suppress(ImportError):
                # pylint: disable=unused-import
                import openvino  # noqa: F401

                backend = "openvino"
        if backend.lower() == "auto":
            backend = "onnx"

        self._call_fn, inputs, outputs = _get_eval_backend(backend.lower(), model)
        self.inputs = inputs
        self.outputs = outputs

    def __call__(
        self,
        output_names: Sequence[str] | None,
        feed_inputs: Dict[str, np.ndarray] | None = None,
        **kw_feeds: np.ndarray,
    ) -> List[np.ndarray]:
        if feed_inputs is None and not kw_feeds:
            raise ValueError(
                "Either feed_inputs or **kw_feeds should be provided."
                " But both are None."
            )
        model_inputs: Dict[str, np.ndarray] = {}
        if feed_inputs:
            model_inputs.update(feed_inputs)
        if kw_feeds:
            model_inputs.update(kw_feeds)
        return self._call_fn(output_names, model_inputs)
