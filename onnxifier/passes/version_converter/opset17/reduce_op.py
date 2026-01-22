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

from typing import List

from onnx.onnx_pb import NodeProto

from .... import OnnxGraph
from ...pattern import SingleNodePattern
from ...rewriter import Rewriter
from . import OP_CONVERTER


class ReduceOp(Rewriter):
    """Move axes from inputs to attributes."""

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        node = nodes[0]
        axes = self.get_value(node.input[1])
        if axes is None:
            raise ValueError(f"axes of {node.name} is not constant")
        self.set_attribute(node, "axes", axes.tolist())
        noop_with_empty_axes = self.get_attribute(node, "noop_with_empty_axes")
        if noop_with_empty_axes:
            raise ValueError(
                f"{self.__class__.__name__} below opset 18 doesn't "
                "support noop_with_empty_axes=1"
            )
        else:
            self.remove_attribute(node, "noop_with_empty_axes")
        node.input.pop(1)
        self += node  # refresh the edge


@OP_CONVERTER.register("ReduceMean")
class ReduceMean(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceMean"))


@OP_CONVERTER.register("ReduceMax")
class ReduceMax(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceMax"))


@OP_CONVERTER.register("ReduceMin")
class ReduceMin(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceMin"))


@OP_CONVERTER.register("ReduceProd")
class ReduceProd(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceProd"))


@OP_CONVERTER.register("ReduceL1")
class ReduceL1(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceL1"))


@OP_CONVERTER.register("ReduceL2")
class ReduceL2(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceL2"))


@OP_CONVERTER.register("ReduceLogSum")
class ReduceLogSum(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceLogSum"))


@OP_CONVERTER.register("ReduceLogSumExp")
class ReduceLogSumExp(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceLogSumExp"))


@OP_CONVERTER.register("ReduceSumSquare")
class ReduceSumSquare(ReduceOp):
    """Move axes from inputs to attributes."""

    def __init__(self):
        super().__init__(SingleNodePattern("ReduceSumSquare"))
