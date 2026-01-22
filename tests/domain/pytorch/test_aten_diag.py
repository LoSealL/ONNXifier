"""
Copyright (C) 2024 The ONNXIFIER Authors.

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

import io

import numpy as np
import onnx
import pytest

from onnxifier.evaluator import Evaluator

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


class Diag(nn.Module):
    def forward(self, x):
        return torch.diag(x, diagonal=0)


def test_diag_export_without_onnxifier():
    """Pytorch has no implementation of torch.diag for onnx."""

    diag = Diag()
    f = io.BytesIO()
    with pytest.raises(RuntimeError):
        torch.onnx.export(diag, (torch.randn(3, 3),), f, opset_version=17, dynamo=False)


def test_diag_export_in_onnxifier():
    """Register aten::diag symbolic by import onnxifier"""
    # pylint: disable=import-outside-toplevel, unused-import
    import onnxifier.domain.pytorch.diag  # noqa: F401

    diag = Diag()
    f = io.BytesIO()
    x = torch.randn(3)
    torch.onnx.export(diag, (x,), f, opset_version=17, dynamo=False)

    f.seek(0)
    model = onnx.load(f)
    assert model.graph.node[0].op_type == "diag"

    runner = Evaluator(model, "onnxruntime")

    y = diag(x).detach().numpy()
    z = runner([], {model.graph.input[0].name: x.numpy()})[0]

    assert np.allclose(y, z)
