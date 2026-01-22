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

Extract onnx functions to individual onnx files.
"""

import argparse
from pathlib import Path
from typing import List

import onnx
from onnx.helper import make_graph, make_model, make_tensor_value_info

from onnxifier import OnnxGraph


def _legal_name(name: str) -> str:
    return name.replace(":", "_").replace("/", "_").replace("\\", "_").replace("?", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    onnx_graph = OnnxGraph(onnx.load_model(args.onnx))
    if not onnx_graph.functions:
        print("No functions found in the onnx model.")
        return 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    for name, func in onnx_graph.functions.items():
        # find nodes use the function
        func_nodes: List[onnx.NodeProto] = []
        for _, node in onnx_graph.nodes.items():
            node_pb: onnx.NodeProto = node["pb"]
            if node_pb.op_type == name and node_pb.domain == func.domain:
                func_nodes.append(node_pb)
        if not func_nodes:
            print(f"No users of function {name}.")
            continue
        print(f"Extracting function {name} to {output_dir / _legal_name(name)}")
        # input spec
        input_spec = []
        for i, input_name in enumerate(func_nodes[0].input):
            shape, etype = onnx_graph.tensor_info(input_name)
            spec = make_tensor_value_info(func.input[i], etype, shape)
            input_spec.append(spec)
        # output spec
        output_spec = []
        for i, output_name in enumerate(func_nodes[0].output):
            shape, etype = onnx_graph.tensor_info(output_name)
            spec = make_tensor_value_info(func.output[i], etype, shape)
            output_spec.append(spec)
        # create graph
        graph = make_graph(
            nodes=func.node,
            name=name,
            inputs=input_spec,
            outputs=output_spec,
            initializer=onnx_graph.initializer,
            value_info=func.value_info,
        )
        # for simplicity, extend functions if func still contains functions
        has_functions = any(
            node.domain and "onnx" not in node.domain for node in func.node
        )  # domain != "" or "ai.onnx"
        if has_functions:
            functions = onnx_graph.functions.values()
        else:
            functions = []
        model = make_model(
            graph,
            model_version=onnx_graph.model_version,
            ir_version=onnx_graph.ir_version,
            functions=functions,
            opset_imports=onnx_graph._model.opset_import,
        )
        onnx.save_model(model, output_dir / f"{_legal_name(name)}.onnx")


if __name__ == "__main__":
    main()
