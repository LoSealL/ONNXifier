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

from ..... import OnnxGraph
from .....passes import PASSES, Registry
from .....passes.pattern import (
    ConstantGraphPattern,
    GraphPattern,
    InputNodePattern,
    OrPattern,
    OutputNodePattern,
    Pattern,
    SingleNodePattern,
    StartEndPointPattern,
)
from .....passes.rewriter import Rewriter, RewriterRepeat
from .....passes.utils import attribute_value, cast_in, cast_out, make_constant

IR_PASSES = Registry("IR_PASS", parent=PASSES)

__all__ = [
    "IR_PASSES",
    "OnnxGraph",
    "ConstantGraphPattern",
    "GraphPattern",
    "InputNodePattern",
    "OrPattern",
    "OutputNodePattern",
    "Pattern",
    "SingleNodePattern",
    "StartEndPointPattern",
    "Rewriter",
    "RewriterRepeat",
    "attribute_value",
    "cast_in",
    "cast_out",
    "make_constant",
]
