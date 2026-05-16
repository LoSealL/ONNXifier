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

import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_operatorsetid,
    make_tensor_value_info,
)

from onnxifier import ONNXIFIER_IR_VERSION, ONNXIFIER_OPSET
from onnxifier.domain.shape_inference import get_shape_inference
from onnxifier.domain.trt.ops.int4_gemm_plugin import (
    int4_gemm_plugin_schema,
    int4_gemm_plugin_shape_infer,
)


def test_int4_gemm_plugin_schema():
    graph = make_graph(
        [
            make_node(
                int4_gemm_plugin_schema.name,
                ["input", "qweight", "scales"],
                ["output"],
                "igp",
                domain=int4_gemm_plugin_schema.domain,
                gemm_n=4096,
                gemm_k=4096,
                group_size=128,
            )
        ],
        "test",
        [
            make_tensor_value_info("input", 1, [1, 4096]),
            make_tensor_value_info("qweight", TensorProto.INT8, [4096, 4096]),
            make_tensor_value_info("scales", TensorProto.FLOAT16, [4096, 4096]),
        ],
        [make_tensor_value_info("output", 1, [1, 4096])],
    )
    model = make_model(
        graph,
        ir_version=ONNXIFIER_IR_VERSION,
        opset_imports=[
            ONNXIFIER_OPSET,
            make_operatorsetid(int4_gemm_plugin_schema.domain, 1),
        ],
    )
    onnx.checker.check_model(model)


def test_int4_gemm_plugin_shape_inference_onnxscript():
    out = int4_gemm_plugin_shape_infer(
        inputs=np.zeros([1, 2048], np.float32),
        qweight=np.zeros([1024, 4096], np.int8),
        scales=np.zeros([1, 128], np.float16),
        gemm_n=4096,
        gemm_k=2048,
        group_size=128,
    )
    assert out.shape == (1, 4096)

    onnxfunc = get_shape_inference(
        int4_gemm_plugin_schema.domain, int4_gemm_plugin_schema.name
    )
    assert onnxfunc is not None
    assert onnxfunc.domain == int4_gemm_plugin_schema.domain
    assert onnxfunc.name == int4_gemm_plugin_schema.name
    assert len(onnxfunc.input) == len(int4_gemm_plugin_schema.inputs)
    assert len(onnxfunc.output) == len(int4_gemm_plugin_schema.outputs)
