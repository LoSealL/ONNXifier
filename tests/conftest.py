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


def classification_models():
    """A list of testing classification models from onnx hub."""
    header = "https://github.com/onnx/models/raw/main/validated/vision/classification/"
    models = [
        ("mobilenet/model/mobilenetv2-10.onnx", None),
        ("resnet/model/resnet18-v2-7.onnx", None),
        ("resnet/model/resnet50-v1-12-qdq.onnx", None),
    ]
    return [(header + i + "?download=", hash_value) for (i, hash_value) in models]


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Generate parametrized arguments to all tests with arg 'model'.
    `model` is acquired from model_zoo.
    """

    if "classification_models" in metafunc.fixturenames:
        metafunc.parametrize("classification_models", classification_models())
