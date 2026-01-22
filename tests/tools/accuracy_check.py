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

import argparse
from pathlib import Path

import numpy as np

from onnxifier import convert
from onnxifier.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("-i", "--image", nargs="+")
    parser.add_argument(
        "-a",
        "--activate",
        nargs="*",
        help="select passes to be activated, activate all passes if not set.",
    )
    parser.add_argument(
        "-b1",
        "--backend",
        default="OpenVINO",
        choices=["onnx", "OpenVINO", "OnnxRuntime"],
        help="select a backend to run the original model.",
    )
    parser.add_argument(
        "-b2",
        "--reference-backend",
        default="OnnxRuntime",
        choices=["onnx", "OpenVINO", "OnnxRuntime"],
        help="select a backend to run the converted model.",
    )
    args = parser.parse_args()

    ori_model = Path(args.model).expanduser()
    model = convert(ori_model, args.activate, strict=False)

    ori_runner = Evaluator(str(ori_model), backend=args.backend)
    ort_runner = Evaluator(model, backend=args.reference_backend)

    ori_inputs = {}
    for _, (name, info) in enumerate(ori_runner.inputs.items()):
        try:
            dtype = np.iinfo(info["dtype"]).dtype
            # all zeros for integer types
            ori_inputs[name] = np.zeros(info["shape"], dtype)
        except ValueError:
            ori_inputs[name] = np.random.rand(*info["shape"]).astype(info["dtype"])
    ori_results = ori_runner(sorted(ori_runner.outputs.keys()), ori_inputs)

    ort_inputs = {}
    for name, info in ort_runner.inputs.items():
        if name in ori_inputs:
            ort_inputs[name] = ori_inputs[name]
        else:
            found = False
            for j in ori_inputs.values():
                if tuple(j.shape) == info["shape"]:
                    ort_inputs[name] = j
                    found = True
                    break
            if not found:
                raise ValueError(f"Cannot find input {name} in converted model")
    ort_results = ort_runner(sorted(ort_runner.outputs.keys()), ort_inputs)

    errors = {}
    for name1, name2, ov_ans, ort_ans in zip(
        sorted(ori_runner.outputs), sorted(ort_runner.outputs), ori_results, ort_results
    ):
        if ov_ans.shape != ort_ans.shape:
            raise ValueError(
                "Output shapes do not match: "
                f"{name1}({ov_ans.shape}) vs {name2}({ort_ans.shape})"
            )
        err = np.abs(ov_ans - ort_ans)
        errors[f"{name1}-{name2}"] = (err, ort_ans)
    for k, v in errors.items():
        err, ref = v
        rel = err / (np.abs(ref) + 1e-5)
        print(f"{k}: max error = {np.max(err):.4f}, mean error = {np.mean(err):.4f}")
        print(f"{k}: rel max   = {np.max(rel):.4f}, rel mean   = {np.mean(rel):.4f}")


if __name__ == "__main__":
    main()
