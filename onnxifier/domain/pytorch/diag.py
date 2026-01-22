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
import onnxscript
import torch
from onnx.numpy_helper import from_array
from onnxscript.onnx_opset import opset17 as op
from onnxscript.values import Opset

try:  # torch>=2.9
    from torch.onnx._internal.torchscript_exporter.jit_utils import GraphContext
    from torch.onnx._internal.torchscript_exporter.registration import onnx_symbolic
except ImportError:
    from torch.onnx._internal.jit_utils import GraphContext
    from torch.onnx._internal.registration import onnx_symbolic

from . import IR_DOMAIN


@onnxscript.script(Opset(domain=IR_DOMAIN.domain, version=IR_DOMAIN.version))
def diag(x, diagonal: int):
    """Implementation of ``aten::diag`` using ONNX operators.

    Note:

        This implementation only supports 1D input
    """

    shape = op.Shape(x)
    idv = op.ConstantOfShape(shape, value=from_array(np.array([1], np.float32)))
    idv_x = op.CastLike(idv, x)
    idv1d_t = op.Unsqueeze(idv_x, axes=0)
    x_1d = op.Unsqueeze(x, axes=1)
    y = op.MatMul(x_1d, idv1d_t)
    mask = op.EyeLike(y, k=diagonal)
    y = op.Mul(y, mask)
    return y


@onnx_symbolic("aten::diag", 1, custom=True)
@torch.onnx.symbolic_helper.parse_args("v", "i")
def diag_symbolic(g: GraphContext, x: torch._C.Value, diagonal: int = 0):
    """Symbolic function for ``torch.diag``."""

    return g.onnxscript_op(diag, x, diagonal_i=diagonal).setTypeAs(x)
