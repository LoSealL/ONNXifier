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

import warnings
from pathlib import Path

import click
import networkx as nx
import numpy as np
import onnx

from onnxifier import OnnxGraph, PassManager

# from onnx.helper import get_node_attr_value


def conv_analysis(
    graph: OnnxGraph,
    node: onnx.NodeProto,
):
    """Analyze Conv node.

    Args:
        graph (OnnxGraph): Graph.
        node (onnx.NodeProto): Node.

    Returns:
        dict: Analysis result.
    """
    ishape = graph.static_tensor_shape(node.input[0])
    wshape = graph.static_tensor_shape(node.input[1])
    oshape = graph.static_tensor_shape(node.output[0])
    if len(node.input) > 2 and node.input[2] != "":
        bias = graph.static_tensor_shape(node.input[2])
    else:
        bias = None
    # strides = get_node_attr_value(node, "strides")

    ic = ishape[1]
    oc = wshape[1]
    kh = wshape[2]
    kw = wshape[3]
    oh = oshape[2]
    ow = oshape[3]
    return dict(
        layer=node.name,
        op_type=node.op_type,
        ic=ic,
        oc=oc,
        kh=kh,
        kw=kw,
        idepth=1,
        ih=ishape[2],
        iw=ishape[3],
        oh=oh,
        ow=ow,
        act=0,
        order=0,
        ops=oc * oh * ow * (ic * kh * kw + (1 if bias else 0)),
        load=(
            np.prod(ishape) + np.prod(wshape) + (np.prod(bias) if bias else 0)
        ).item(),
        store=np.prod(oshape).item(),
    )


def matmul_analysis(
    graph: OnnxGraph,
    node: onnx.NodeProto,
):
    """Analyze MatMul node.

    Args:
        graph (OnnxGraph): Graph.
        node (onnx.NodeProto): Node.

    Returns:
        dict: Analysis result.
    """
    shape_a = graph.static_tensor_shape(node.input[0])
    shape_b = graph.static_tensor_shape(node.input[1])

    shape_a = [np.prod(shape_a[:-1]), shape_a[-1]]
    shape_b = [shape_b[0], np.prod(shape_b[1:])]
    m, n, k = shape_a[0], shape_b[1], shape_a[1]

    return dict(
        layer=node.name,
        op_type=node.op_type,
        ic=m,
        oc=n,
        kh=1,
        kw=1,
        idepth=1,
        ih=1,
        iw=k,
        oh=1,
        ow=k,
        act=0,
        order=0,
        ops=m * n * k,
        load=(np.prod(shape_a) + np.prod(shape_b)).item(),
        store=(m * n).item(),
    )


def unary_analysis(
    graph: OnnxGraph,
    node: onnx.NodeProto,
    no_ops: bool = False,
):
    """Analyze unary (activation) node.

    Args:
        graph (OnnxGraph): Graph.
        node (onnx.NodeProto): Node.

    Returns:
        dict: Analysis result.
    """
    ishape = graph.static_tensor_shape(node.input[0])
    oshape = graph.static_tensor_shape(node.output[0])
    while len(ishape) < 4:
        ishape.append(1)
    while len(oshape) < 4:
        oshape.append(1)
    return dict(
        layer=node.name,
        op_type=node.op_type,
        ic=ishape[1],
        oc=ishape[1],
        kh=1,
        kw=1,
        idepth=1,
        ih=ishape[2],
        iw=ishape[3],
        oh=oshape[2],
        ow=oshape[3],
        act=1,
        order=0,
        ops=0 if no_ops else np.prod(ishape).item(),
        load=np.prod(ishape).item(),
        store=np.prod(oshape).item(),
    )


def binary_analysis(
    graph: OnnxGraph,
    node: onnx.NodeProto,
):
    """Analyze Binary node.

    Args:
        graph (OnnxGraph): Graph.
        node (onnx.NodeProto): Node.

    Returns:
        dict: Analysis result.
    """
    shape = graph.static_tensor_shape(node.output[0])
    n_inputs = len(node.input)
    while len(shape) < 4:
        shape.append(1)

    return dict(
        layer=node.name,
        op_type=node.op_type,
        ic=shape[1],
        oc=shape[1],
        kh=1,
        kw=1,
        idepth=1,
        ih=shape[2],
        iw=shape[3],
        oh=shape[2],
        ow=shape[3],
        act=0,
        order=1,
        ops=np.prod(shape).item(),
        load=np.prod(shape).item() * n_inputs,
        store=np.prod(shape).item(),
    )


def resize_analysis(
    graph: OnnxGraph,
    node: onnx.NodeProto,
):
    """Analyze Resize node.

    Args:
        graph (OnnxGraph): Graph.
        node (onnx.NodeProto): Node.

    Returns:
        dict: Analysis result.
    """
    ishape = graph.static_tensor_shape(node.input[0])
    oshape = graph.static_tensor_shape(node.output[0])

    ic = ishape[1]
    oc = oshape[1]
    ih = ishape[2]
    iw = ishape[3]
    oh = oshape[2]
    ow = oshape[3]
    return dict(
        layer=node.name,
        op_type=node.op_type,
        ic=ic,
        oc=oc,
        kh=oh // ih,
        kw=ow // iw,
        idepth=1,
        ih=ih,
        iw=iw,
        oh=oh,
        ow=ow,
        act=0,
        order=1,
        ops=np.prod(oshape).item() * 4,
        load=np.prod(ishape).item(),
        store=np.prod(oshape).item(),
    )


def pool_analysis(
    graph: OnnxGraph,
    node: onnx.NodeProto,
):
    """Analyze Pool node.

    Args:
        graph (OnnxGraph): Graph.
        node (onnx.NodeProto): Node.

    Returns:
        dict: Analysis result.
    """

    ishape = graph.static_tensor_shape(node.input[0])
    oshape = graph.static_tensor_shape(node.output[0])

    ic = ishape[1]
    oc = oshape[1]
    ih = ishape[2]
    iw = ishape[3]
    oh = oshape[2]
    ow = oshape[3]
    return dict(
        layer=node.name,
        op_type=node.op_type,
        ic=ic,
        oc=oc,
        kh=oh // ih,
        kw=ow // iw,
        idepth=1,
        ih=ih,
        iw=iw,
        oh=oh,
        ow=ow,
        act=0,
        order=1,
        ops=np.prod(oshape).item(),
        load=np.prod(ishape).item(),
        store=np.prod(oshape).item(),
    )


def gridsample_analysis(
    graph: OnnxGraph,
    node: onnx.NodeProto,
):
    """Analyze GridSample node.

    Args:
        graph (OnnxGraph): Graph.
        node (onnx.NodeProto): Node.

    Returns:
        dict: Analysis result.
    """

    ishape = graph.static_tensor_shape(node.input[0])
    oshape = graph.static_tensor_shape(node.output[0])
    grid_shape = graph.static_tensor_shape(node.input[1])
    while len(ishape) < 5:
        ishape.append(1)
    while len(oshape) < 5:
        oshape.append(1)
    return dict(
        layer=node.name,
        op_type=node.op_type,
        ic=ishape[1],
        oc=oshape[1],
        kh=1,
        kw=1,
        idepth=ishape[-1],
        ih=ishape[2],
        iw=ishape[3],
        oh=oshape[2],
        ow=oshape[3],
        act=0,
        order=1,
        ops=4 * np.prod(grid_shape).item() * ishape[1],
        load=4 * np.prod(grid_shape).item() * ishape[1],
        store=np.prod(oshape).item(),
    )


def scatternd_analysis(
    graph: OnnxGraph,
    node: onnx.NodeProto,
):
    """Analyze ScatterND node.

    Args:
        graph (OnnxGraph): Graph.
        node (onnx.NodeProto): Node.

    Returns:
        dict: Analysis result.
    """

    update_shape = graph.static_tensor_shape(node.input[2])
    return dict(
        layer=node.name,
        op_type=node.op_type,
        ic=1,
        oc=1,
        kh=1,
        kw=1,
        idepth=1,
        ih=1,
        iw=1,
        oh=1,
        ow=1,
        act=0,
        order=1,
        ops=0,
        load=np.prod(update_shape).item(),
        store=np.prod(update_shape).item(),
    )


def gather_analysis(
    graph: OnnxGraph,
    node: onnx.NodeProto,
):
    """Analyze Gather node.

    Args:
        graph (OnnxGraph): Graph.
        node (onnx.NodeProto): Node.

    Returns:
        dict: Analysis result.
    """
    oshape = graph.static_tensor_shape(node.output[0])
    return dict(
        layer=node.name,
        op_type=node.op_type,
        ic=1,
        oc=1,
        kh=1,
        kw=1,
        idepth=1,
        ih=1,
        iw=1,
        oh=1,
        ow=1,
        act=0,
        order=1,
        ops=0,
        load=np.prod(oshape).item(),
        store=np.prod(oshape).item(),
    )


@click.command()
@click.option("-o", "--output", type=click.Path(dir_okay=False), required=True)
@click.argument(
    "model",
    type=click.Path(exists=True, dir_okay=False),
)
def analysis(model: None | Path = None, output: None | Path = None):
    assert model is not None
    graph = OnnxGraph(onnx.load_model(model))
    graph = PassManager(["infer_shape"]).optimize(graph)

    data = []
    for node_name in nx.topological_sort(graph):
        node: onnx.NodeProto = graph.nodes[node_name]["pb"]
        if node.op_type in ("Conv", "ConvTranspose"):
            data.append(conv_analysis(graph, node))
        elif node.op_type in ("MatMul",):
            data.append(matmul_analysis(graph, node))
        elif node.op_type in (
            "Relu",
            "Tanh",
            "Sigmoid",
            "HardSwish",
            "HardSigmoid",
            "Erf",
            "Transpose",
            "Sqrt",
            "Pow",
            "Clip",
        ):
            data.append(unary_analysis(graph, node))
        elif node.op_type in ("Tile",):
            data.append(unary_analysis(graph, node, no_ops=True))
        elif node.op_type in ("Add", "Sub", "Mul", "Div"):
            data.append(binary_analysis(graph, node))
        elif node.op_type in ("Resize",):
            data.append(resize_analysis(graph, node))
        elif node.op_type in ("MaxPool", "AveragePool"):
            data.append(pool_analysis(graph, node))
        elif node.op_type in ("GridSample",):
            data.append(gridsample_analysis(graph, node))
        elif node.op_type in ("ScatterND",):
            data.append(scatternd_analysis(graph, node))
        elif node.op_type in ("Gather", "GatherND", "GatherElements"):
            data.append(gather_analysis(graph, node))
        else:
            warnings.warn(f"Not support {node.op_type} now.")
    assert output is not None
    with open(output, "w", encoding="utf-8") as f:
        f.write(",".join(data[0].keys()) + "\n")
        for row in data:
            f.write(",".join(map(str, row.values())) + "\n")


if __name__ == "__main__":
    analysis()
