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

# check onnxscript and torch installation

try:
    import torch

    if torch.__version__ < "2.0.0":
        raise ImportError("PyTorch version should be 2.0.0 or higher.")
except ImportError as e:
    raise ImportError("PyTorch>=2.0.0 is not installed.") from e


try:
    import onnxscript
except ImportError as e:
    raise ImportError("onnxscript is not installed.") from e


from onnx import OperatorSetIdProto
from onnx.helper import make_operatorsetid

IR_DOMAIN: OperatorSetIdProto = make_operatorsetid("ai.pytorch.aten", 2)
"""Pytorch ATen"""
