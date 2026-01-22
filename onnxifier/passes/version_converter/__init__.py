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

from ...passes import Registry
from ...traits import RewriterInterface

OP_CONVERTER_19 = Registry[RewriterInterface]("op_converter_19")
OP_CONVERTER_17 = Registry[RewriterInterface]("op_converter_17")
OP_CONVERTER_15 = Registry[RewriterInterface]("op_converter_15")
OP_CONVERTER_13 = Registry[RewriterInterface]("op_converter_13")
OP_CONVERTER = {
    13: OP_CONVERTER_13,
    15: OP_CONVERTER_15,
    17: OP_CONVERTER_17,
    19: OP_CONVERTER_19,
}
