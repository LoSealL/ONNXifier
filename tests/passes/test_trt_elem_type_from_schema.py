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

import pytest
from onnx import TensorProto

from onnxifier.domain.trt.ops.mamba_plugin import causal_conv1d_schema
from onnxifier.domain.trt.ops.recurrent_plugin import gated_delta_rule_schema
from onnxifier.passes.swap.tensorrt import elem_type_from_schema


@pytest.mark.parametrize(
    ["schema", "expect"],
    [
        (
            causal_conv1d_schema,
            [
                TensorProto.FLOAT16,
                TensorProto.FLOAT16,
                TensorProto.FLOAT16,
                TensorProto.FLOAT16,
                TensorProto.INT32,
            ],
        ),
        (
            gated_delta_rule_schema,
            [
                TensorProto.FLOAT16,
                TensorProto.FLOAT16,
                TensorProto.FLOAT16,
                TensorProto.FLOAT16,
                TensorProto.FLOAT16,
                TensorProto.FLOAT16,
                TensorProto.INT32,
            ],
        ),
    ],
)
def test_get_elem_type_from_schema(schema, expect):
    for i, e in zip(schema.inputs, expect):
        dtype = elem_type_from_schema(schema, i.name)
        assert dtype == e
