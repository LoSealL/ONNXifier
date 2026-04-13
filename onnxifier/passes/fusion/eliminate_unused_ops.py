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

# pylint: disable=arguments-differ

from onnx.onnx_pb import NodeProto

from ... import OnnxGraph
from .. import L2
from ..pattern import SingleNodePattern
from ..rewriter import Rewriter


def _bypass(rewriter: Rewriter, graph: OnnxGraph, node: NodeProto) -> None:
    """Redirect all uses of ``node.output[0]`` to ``node.input[0]``,
    then schedule the node for removal.

    When the node's output is a graph-level output the predecessor's
    output tensor is renamed to preserve the required output name.
    When the node has no predecessor (its data input is a graph
    input) the graph-output case is skipped for safety.
    """
    source = node.input[0]  # data flowing *into* the no-op
    target = node.output[0]  # name we want to stop producing

    if graph.nodes[node.name]["has_output"]:
        predecessors = graph.onnx_predecessors(node)
        if not predecessors:
            # Source is a direct graph input — leave graph outputs intact.
            return
        prev = predecessors[0]
        # Rename the predecessor's output from source → target so
        # downstream consumers keep working with the graph-output name.
        prev_changed = False
        for i, out in enumerate(prev.output):
            if out == source:
                prev.output[i] = target
                prev_changed = True
        # Update sibling nodes that also consumed source.
        for sibling in graph.onnx_siblings(node):
            sibling_changed = False
            for i, inp in enumerate(sibling.input):
                if inp == source:
                    sibling.input[i] = target
                    sibling_changed = True
            if sibling_changed:
                # Need to rebuild edges after node input rewrite.
                rewriter += sibling
        if prev_changed:
            # Need to rebuild producer/output mapping after output rewrite.
            rewriter += prev
    else:
        # Rewire every downstream user from target → source.
        for succ in graph.onnx_successors(node):
            succ_changed = False
            for i, inp in enumerate(succ.input):
                if inp == target:
                    succ.input[i] = source
                    succ_changed = True
            if succ_changed:
                # Need to rebuild edges after node input rewrite.
                rewriter += succ

    rewriter -= node


@L2.register(name="eliminate_nop_slice", deps=["infer_shape"])
class EliminateNopSliceRewriter(Rewriter):
    """Remove no-op Slice/Split nodes.

    - Slice: every sliced axis covers full extent
      (e.g. ``a[:100]`` when ``shape(a) == [16]``).
    - Split: only one output and split covers the whole selected axis.
    """

    def __init__(self):
        super().__init__(
            pattern=SingleNodePattern("Slice") | SingleNodePattern("Split")
        )

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto]):
        node = nodes[0]
        if node.op_type == "Split":
            self._try_eliminate_split(graph, node)
            return
        self._try_eliminate_slice(graph, node)

    def _try_eliminate_slice(self, graph: OnnxGraph, node: NodeProto) -> None:
        # Inputs: data(0), starts(1), ends(2), [axes(3)], [steps(4)]
        try:
            data_shape = graph.tensor_shape(node.input[0])
        except ValueError:
            return

        starts_val = self.get_value(node.input[1])
        ends_val = self.get_value(node.input[2])
        if starts_val is None or ends_val is None:
            return

        starts = [int(v) for v in starts_val.flatten()]
        ends = [int(v) for v in ends_val.flatten()]
        rank = len(data_shape)

        # Resolve axes (defaults to 0 … len(starts)-1 when not provided).
        if len(node.input) > 3 and node.input[3]:
            axes_val = self.get_value(node.input[3])
            if axes_val is None:
                return
            axes = [int(a) % rank for a in axes_val.flatten()]
        else:
            axes = list(range(len(starts)))

        # Resolve steps (defaults to 1 when not provided).
        if len(node.input) > 4 and node.input[4]:
            steps_val = self.get_value(node.input[4])
            if steps_val is None:
                return
            steps = [int(s) for s in steps_val.flatten()]
        else:
            steps = [1] * len(starts)

        # Each sliced axis must start at 0, end at (or beyond) dim, step by 1.
        for i, axis in enumerate(axes):
            dim = data_shape[axis]
            if not isinstance(dim, int) or dim <= 0:
                return  # dynamic or unknown dimension — cannot verify statically

            start, end, step = starts[i], ends[i], steps[i]

            # Clamp per ONNX Slice specification.
            if start < 0:
                start = max(0, dim + start)
            else:
                start = min(start, dim)
            if end < 0:
                end = max(0, dim + end)
            else:
                end = min(end, dim)

            if step != 1 or start != 0 or end < dim:
                return  # not a full-coverage slice on this axis

        _bypass(self, graph, node)

    def _try_eliminate_split(self, graph: OnnxGraph, node: NodeProto) -> None:
        # Split can only be bypassed when it has a single output.
        if len(node.output) != 1:
            return

        try:
            input_shape = graph.tensor_shape(node.input[0])
        except ValueError:
            return

        axis = self.get_attribute(node, "axis", 0)
        if not isinstance(axis, int):
            return
        rank = len(input_shape)
        axis %= rank
        dim = input_shape[axis]
        if not isinstance(dim, int) or dim <= 0:
            return

        # split from 2nd input (newer opsets) has higher priority.
        split: list[int] | None = None
        if len(node.input) > 1 and node.input[1]:
            split_val = self.get_value(node.input[1])
            if split_val is None:
                return
            split = [int(v) for v in split_val.flatten()]
        else:
            split_attr = self.get_attribute(node, "split")
            if split_attr is not None:
                split = [int(v) for v in split_attr]

        if split is None:
            # Split without explicit split and only one output is a pass-through.
            _bypass(self, graph, node)
            return
        if len(split) != 1:
            return

        # Accept oversized split as no-op intent on single-output split.
        if split[0] >= dim:
            _bypass(self, graph, node)


@L2.register(name="eliminate_nop_concat", deps=["infer_shape"])
class EliminateSingleConcatRewriter(Rewriter):
    """Remove Concat nodes that have exactly one active input."""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Concat"))

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto]):
        node = nodes[0]
        active = [inp for inp in node.input if inp]
        if len(active) == 1:
            _bypass(self, graph, node)


@L2.register(name="eliminate_nop_transpose", deps=["infer_shape"])
class EliminateIdentityTransposeRewriter(Rewriter):
    """Remove Transpose nodes whose permutation is the identity ``[0, 1, …, n-1]``."""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Transpose"))

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto]):
        node = nodes[0]
        perm = self.get_attribute(node, "perm")
        if perm is None:
            # No perm attribute: ONNX defaults to reversing all axes.
            # That is identity only for rank ≤ 1.
            try:
                shape = graph.tensor_shape(node.input[0])
            except ValueError:
                return
            if len(shape) <= 1:
                _bypass(self, graph, node)
            return

        if list(perm) == list(range(len(perm))):
            _bypass(self, graph, node)


@L2.register(name="eliminate_nop_pad", deps=["infer_shape"])
class EliminateZeroPadRewriter(Rewriter):
    """Remove Pad nodes where every padding value is zero."""

    def __init__(self):
        super().__init__(pattern=SingleNodePattern("Pad"))

    def rewrite(self, graph: OnnxGraph, nodes: list[NodeProto]):
        node = nodes[0]
        if len(node.input) < 2 or not node.input[1]:
            return
        pads_val = self.get_value(node.input[1])
        if pads_val is None:
            return
        if all(int(p) == 0 for p in pads_val.flatten()):
            _bypass(self, graph, node)
