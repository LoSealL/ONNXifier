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

Remove conv weights with all zero channels
"""

from itertools import chain
from typing import Dict, List, Protocol, Set

import networkx as nx
import numpy as np
from onnx.onnx_pb import NodeProto

from ... import OnnxGraph, logger
from ...algo.subgraph import find_inbound_nodes, find_outbound_nodes
from .. import PASSES, Registry
from ..pattern import Pattern, SingleNodePattern
from ..rewriter import Rewriter
from ..utils import make_constant

_CHAN_MIXER = ("Conv", "ConvTranspose")
_WHITELIST = (
    "Cast",
    "Clip",
    "Erf",
    "Gelu",
    "HardSigmoid",
    "HardSwish",
    "LeakyRelu",
    "MaxPool",
    "Mish",
    "PRelu",
    "Relu",
    "Sigmoid",
    "Swish",
    "Tanh",
)
_BLACKLIST = ()


class PrunedData:
    """Helper class to store pruned data."""

    def __init__(self):
        # mapping node to its output channel indices of zero values
        # E.g:
        # Y = Conv(X, W), Y[:, ind] == 0
        # out_zero_indices[Y] = {ind}
        self.fw_zero_indices: Dict[NodeProto, Set[int]] = {}
        # mapping node's predecessor to this node's input channel indices of zero values
        # E.g:
        # Y = Conv(X, W), X[:, ind] == 0
        # in_zero_indices[X] = {ind}
        # Note: input channels will be merged if two nodes share the same input
        self.bw_zero_indices: Dict[NodeProto, Set[int]] = {}
        self.bw_stopper: Dict[NodeProto, bool] = {}
        self.axes_map: Dict[NodeProto, Set[int]] = {}
        self.seen_nodes: Set[str] = set()


class IPruneForwardFunc(Protocol):
    """Interface for forward pruning functions."""

    __DEPS__: List[str]
    __PATCHES__: List[str]

    def __call__(
        self, rewriter: Rewriter, node: NodeProto, pruned: PrunedData
    ) -> Set[int]:
        return set()


class IPruneBackwardFunc(Protocol):
    """Interface for backward pruning functions."""

    __DEPS__: List[str]
    __PATCHES__: List[str]

    def __call__(
        self, rewriter: Rewriter, node: NodeProto, pruned: PrunedData
    ) -> List[Set[int]]:
        return []


FW_PRUNE = Registry[IPruneForwardFunc]("ForwardPruner")
BW_PRUNE = Registry[IPruneBackwardFunc]("BackwardPruner")
MIXTURE = Registry[IPruneForwardFunc]("Mixture")


def _pass_through_forward(
    self: Rewriter,
    node: NodeProto,
    pruned: PrunedData,
):
    pred = self.graph.onnx_predecessors(node)[0]
    return pruned.fw_zero_indices[pred]


def _pass_through_backward(
    self: Rewriter,  # pylint: disable=unused-argument
    node: NodeProto,
    pruned: PrunedData,
):
    return [pruned.bw_zero_indices[node]]


def _default_mixture(
    self: Rewriter,  # pylint: disable=unused-argument
    node: NodeProto,
    pruned: PrunedData,
):
    return pruned.fw_zero_indices[node]


for op_type in _WHITELIST:
    FW_PRUNE.register(op_type)(_pass_through_forward)
    BW_PRUNE.register(op_type)(_pass_through_backward)


def _get_weights(rewriter: Rewriter, node: NodeProto) -> np.ndarray:
    if node.op_type in ("Conv", "ConvTranspose"):
        weights = rewriter.get_value(node.input[1])
    else:
        raise ValueError(f"Unsupported op_type: {node.op_type}")

    if weights is None:
        raise ValueError(f"Non constant weights of node: {node.name}")
    return weights


def _get_zero_indices(value: np.ndarray, axis: int = 0) -> Set[int]:
    zero_indices = set()
    for i, val in enumerate(np.split(value, value.shape[axis], axis=axis)):
        if np.all(val == 0):
            zero_indices.add(i)
    return zero_indices


@FW_PRUNE.register("Conv")
def get_conv_zero_out_channels(
    self: Rewriter,
    node: NodeProto,
    pruned: PrunedData,  # pylint: disable=unused-argument
) -> Set[int]:
    """Get zeroed output channels of a Conv operator.

    A zeroed output channel ``i`` is a channel that all weights of ``W[i][:]``
    are zeros. If the Conv has bias, the corresponding bias value must be zero
    too.

    Args:
        self (Rewriter): used in class object :class:`RemoveZeroChannelsRewriter`
        node (NodeProto): the Conv operator to be checked
        pruned (PrunedData): a helper dictionary to store the pruned channels
    """
    weights = _get_weights(self, node)
    if group := self.get_attribute(node, "group"):
        if group and group != 1 and weights.shape[1] != 1:
            # not a depthwise conv but with group > 1
            return set()

    oi_w = np.abs(weights).sum(axis=(-1, -2))
    zero_indices = _get_zero_indices(oi_w)
    if len(node.input) > 2:
        bias = self.get_value(node.input[2])
        if bias is None:
            raise ValueError(f"Cannot get a constant value of bias: {node.name}")
        zero_indices.intersection_update(_get_zero_indices(bias))
    return zero_indices


@FW_PRUNE.register("ConvTranspose")
def get_convtranspose_zero_out_channels(
    self: Rewriter,
    node: NodeProto,
    pruned: PrunedData,  # pylint: disable=unused-argument
) -> Set[int]:
    """Get zeroed output channels of a ConvTranspose operator.

    A zeroed output channel ``i`` is a channel that all weights of ``W[:][i]``
    are zeros. If the ConvTranspose has bias, the corresponding bias value must
    be zero too.

    Args:
        self (Rewriter): used in class object :class:`RemoveZeroChannelsRewriter`
        node (NodeProto): the ConvTranspose operator to be checked
        pruned (PrunedData): a helper dictionary to store the pruned channels
    """
    weights = _get_weights(self, node)
    if group := self.get_attribute(node, "group"):
        if group and group != 1 and weights.shape[1] != 1:
            # not a depthwise deconv but with group > 1
            return set()

    io_w = np.abs(weights).sum(axis=(-1, -2))
    zero_indices = _get_zero_indices(io_w, axis=1)
    if len(node.input) > 2:
        bias = self.get_value(node.input[2])
        if bias is None:
            raise ValueError(f"Cannot get a constant value of bias: {node.name}")
        zero_indices.intersection_update(_get_zero_indices(bias))
    return zero_indices


@FW_PRUNE.register("Constant")
def get_constant_zero_out_channels(
    self: Rewriter,
    node: NodeProto,
    pruned: PrunedData,  # pylint: disable=unused-argument
) -> Set[int]:
    """Get zeroed output channels of a constant operator.

    A zeroed output channel ``i`` is a channel that all values are zeros.

    Note:
        The channel axis is fixed to 1.

    Args:
        self (Rewriter): used in class object :class:`RemoveZeroChannelsRewriter`
        node (NodeProto): the Constant operator to be checked
        pruned (PrunedData): unused parameter
    """
    value = self.get_value(node)
    assert value is not None
    zero_indices = _get_zero_indices(value, axis=1)
    return zero_indices


# not ready
def get_add_zero_out_channels(
    self: Rewriter, node: NodeProto, pruned: PrunedData
) -> Set[int]:
    """Get zeroed output channels of an Add or Sub operator.

    A zeroed output channel ``i`` is a channel that all values from both
    operands are zeros.

    Note:
        The channel axis is fixed to 1.

    Args:
        self (Rewriter): used in class object :class:`RemoveZeroChannelsRewriter`
        node (NodeProto): the Add/Sub operator to be checked
        pruned (dict): a helper dictionary to store the pruned channels
    """
    operand_0 = self.get_input_node(node, 0)
    operand_1 = self.get_input_node(node, 1)
    if operand_0 is None:
        value = self.get_value(node.input[0])
        assert value is not None
        indices_0 = _get_zero_indices(value, axis=1)
        axis_0 = value.shape[1]
    else:
        indices_0 = pruned.fw_zero_indices[operand_0]
        axis_0 = self.graph.tensor_shape(node.input[0])[1]
    if operand_1 is None:
        value = self.get_value(node.input[1])
        assert value is not None
        indices_1 = _get_zero_indices(value, axis=1)
        axis_1 = value.shape[1]
    else:
        indices_1 = pruned.fw_zero_indices[operand_1]
        axis_1 = self.graph.tensor_shape(node.input[1])[1]

    if axis_0 == axis_1:
        zero_indices = indices_0.intersection(indices_1)
    elif axis_0 == 1:
        zero_indices = indices_1 if indices_0 else set()
    elif axis_1 == 1:
        zero_indices = indices_0 if indices_1 else set()
    else:
        raise ValueError(f"Broadcast error executing node: {node.name}")
    return zero_indices


@FW_PRUNE.register("Concat")
def get_concat_zero_out_channels(
    self: Rewriter, node: NodeProto, pruned: PrunedData
) -> Set[int]:
    """Get zeroed output channels of a Concat operator.

    A zeroed output channel ``i`` is a channel that corresponding values from one
    input are all zeros.

    Note:
        The channel axis is fixed to 1.

    Args:
        self (Rewriter): used in class object :class:`RemoveZeroChannelsRewriter`
        node (NodeProto): the Concat operator to be checked
        pruned (PrunedData): a helper dictionary to store the pruned channels
    """
    indices_list = []
    ch_axis = 1
    channels = 0
    for i, inp in enumerate(node.input):
        operand = self.get_input_node(node, i)
        if operand is None:
            value = self.get_value(node.input[i])
            assert value is not None
            indices = _get_zero_indices(value, axis=1)
        else:
            indices = pruned.fw_zero_indices[operand]
        indices_list.extend([i + channels for i in indices])
        shape = self.graph.tensor_shape(inp)[ch_axis]
        assert isinstance(shape, int)
        channels += shape
    zero_indices = set(indices_list)
    return zero_indices


@BW_PRUNE.register("Conv")
def get_conv_zero_in_channels(
    self: Rewriter,
    node: NodeProto,
    pruned: PrunedData,  # pylint: disable=unused-argument
) -> List[Set[int]]:
    """Get zeroed input channels of a Conv operator.

    A zeroed input channel ``i`` is a channel that all weights of ``W[:,i]`` are
    zeros.

    Args:
        self (Rewriter): used in class object :class:`RemoveZeroChannelsRewriter`
        node (NodeProto): the Conv operator to be checked
        pruned (PrunedData): unused parameter
    """
    weights = _get_weights(self, node)
    if group := self.get_attribute(node, "group"):
        if group and group != 1 and weights.shape[1] != 1:
            return [set()]

    oi_w = np.abs(weights).sum(axis=(-1, -2))
    zero_indices = _get_zero_indices(oi_w, axis=1)
    return [zero_indices]


@BW_PRUNE.register("ConvTranspose")
def get_convtranspose_zero_in_channels(
    self: Rewriter,
    node: NodeProto,
    pruned: PrunedData,  # pylint: disable=unused-argument
) -> List[Set[int]]:
    """Get zeroed input channels of a ConvTranspose operator.

    A zeroed input channel ``i`` is a channel that all weights of ``W[i,:]`` are
    zeros.

    Args:
        self (Rewriter): used in class object :class:`RemoveZeroChannelsRewriter`
        node (NodeProto): the ConvTranspose operator to be checked
        pruned (PrunedData): unused parameter
    """
    weights = _get_weights(self, node)
    if group := self.get_attribute(node, "group"):
        if group and group != 1 and weights.shape[1] != 1:
            return [set()]

    io_w = np.abs(weights).sum(axis=(-1, -2))
    zero_indices = _get_zero_indices(io_w, axis=0)
    return [zero_indices]


@BW_PRUNE.register("Concat")
def get_concat_zero_in_channels(
    self: Rewriter, node: NodeProto, pruned: PrunedData
) -> List[Set[int]]:
    """Get zeroed input channels of a Concat operator.

    A zeroed input channel ``i`` is a channel that corresponding values from
    output are all zeros.

    Args:
        self (Rewriter): used in class object :class:`RemoveZeroChannelsRewriter`
        node (NodeProto): the Concat operator to be checked
        pruned (PrunedData): a helper dictionary to store the pruned channels
    """
    axes_out = np.asarray(tuple(pruned.bw_zero_indices[node]))
    ch_axis = 1
    zero_indices = []
    for _, inp in enumerate(node.input):
        channels = self.graph.tensor_shape(inp)[ch_axis]
        assert isinstance(channels, int)
        zero_indices.append(set(axes_out[axes_out < channels]))
        axes_out = axes_out[axes_out >= channels] - channels
    return zero_indices


# not ready
def get_add_zero_in_channels(
    self: Rewriter, node: NodeProto, pruned: PrunedData
) -> List[Set[int]]:
    """Get zeroed input channels of an Add or Sub operator.

    A zeroed input channel ``i`` is a channel that corresponding values from both
    operands are all zeros.

    Args:
        self (Rewriter): used in class object :class:`RemoveZeroChannelsRewriter`
        node (NodeProto): the Add/Sub operator to be checked
        pruned (PrunedData): a helper dictionary to store the pruned channels
    """
    axes_out = pruned.bw_zero_indices[node]
    ch_axis = 1
    shape0 = self.graph.tensor_shape(node.input[0])
    shape1 = self.graph.tensor_shape(node.input[1])
    zero_indices: List[Set[int]] = [set(), set()]
    if len(shape0) > ch_axis and (ch := shape0[ch_axis]):
        if isinstance(ch, int) and ch > 1:
            zero_indices[0] = axes_out
    if len(shape1) > ch_axis and (ch := shape1[ch_axis]):
        if isinstance(ch, int) and ch > 1:
            zero_indices[1] = axes_out
    return zero_indices


@MIXTURE.register("Conv")
@MIXTURE.register("ConvTranspose")
def get_conv_mixture(
    self: Rewriter,  # pylint: disable=unused-argument
    node: NodeProto,
    pruned: PrunedData,
) -> Set[int]:
    """Get the mix result of zero channel indices consider both forward and backward
    paths.
    """
    if pruned.bw_stopper.get(node):
        return set()
    # could be output so bw_zero_indices could be empty
    return pruned.fw_zero_indices[node] | pruned.bw_zero_indices.get(node, set())


@PASSES.register("remove_zero_channels", deps=["onnxsim"])
class RemoveZeroChannelsRewriter(Rewriter):
    """Prune zerod channels in constant weights of Conv and ConvTranspose operators."""

    # global temp memory to propagate pruned axes through entire graph
    _memo: Dict[int, PrunedData] = {}

    def __init__(self):
        pattern = sum([SingleNodePattern(op) for op in _CHAN_MIXER])
        assert isinstance(pattern, Pattern)
        super().__init__(pattern=pattern)

    def _find_dep_graph(self, graph: OnnxGraph, node: NodeProto):
        nodes_found = {node}  # type: ignore
        searching_nodes = [(node, "up")]
        while searching_nodes:
            curr_node, direction = searching_nodes.pop(0)
            if direction == "down":
                nodes, endpoints = find_outbound_nodes(graph, curr_node, _CHAN_MIXER)
            else:  # up
                nodes, endpoints = find_inbound_nodes(graph, curr_node, _CHAN_MIXER)
            if new_nodes := (set(endpoints) - nodes_found.intersection(endpoints)):
                direction = "down" if direction == "up" else "up"  # revert direction
                searching_nodes.extend([(op, direction) for op in new_nodes])
            nodes_found.update(nodes + endpoints)
        return tuple(nodes_found)

    def rewrite(self, graph: OnnxGraph, nodes: List[NodeProto], *args, **kwargs):
        if id(graph) not in self._memo:
            self._memo[id(graph)] = PrunedData()
        memo = self._memo[id(graph)]

        # find conv-conv subgraph
        boundary_nodes = self._find_dep_graph(graph, nodes[0])
        if len(boundary_nodes) == 1:
            return
        if any(op.op_type in _BLACKLIST for op in boundary_nodes):
            return

        logger.debug(
            f"Boundary around {nodes[0].name}: {[op.name for op in boundary_nodes]}"
        )
        h = graph.onnx_subgraph(boundary_nodes)
        start_points: List[NodeProto] = []
        for node_name, ind in h.in_degree():  # type: ignore
            if ind == 0 and h.nodes[node_name]["pb"].op_type in _CHAN_MIXER:
                start_points.append(h.nodes[node_name]["pb"])
        logger.debug(f"Start points: {[op.name for op in start_points]}")

        end_points: List[NodeProto] = []
        for node_name, outd in h.out_degree():  # type: ignore
            if outd == 0 and h.nodes[node_name]["pb"].op_type in _CHAN_MIXER:
                end_points.append(h.nodes[node_name]["pb"])
        logger.debug(f"End points: {[op.name for op in end_points]}")

        # get pruned axes
        for n in nx.topological_sort(h):
            self.update_zero_axes_fw(h.nodes[n]["pb"], memo)
        for n in nx.topological_sort(nx.reverse(h, copy=True)):  # BP
            self.update_zero_axes_bw(h.nodes[n]["pb"], memo)
        for i in nx.topological_sort(h):
            refresh_axes_out = self.update_pruned_axes(h.nodes[i]["pb"], memo)
            if refresh_axes_out:
                # refresh all downstreaming nodes of i
                for j in nx.topological_sort(nx.subgraph(h, nx.descendants(h, i))):
                    self.update_zero_axes_fw(h.nodes[j]["pb"], memo)

        for op in chain(start_points, end_points):
            self.remove_weights_channels(op, memo)

    def update_zero_axes_fw(self, node: NodeProto, pruned: PrunedData):
        """Get pruned axis of node in topological order and store in
        :class:`PrunedData`.
        """

        if get_zero_channels := FW_PRUNE.get(node.op_type):
            zero_indices = get_zero_channels(self, node, pruned)
            pruned.fw_zero_indices[node] = zero_indices
            logger.debug(f"zero out channels of {node.name}: {zero_indices}")
        else:
            pruned.fw_zero_indices[node] = set()

    def update_zero_axes_bw(self, node: NodeProto, pruned: PrunedData):
        """Get pruned axis of node in reverse topological order and store in
        :class:`PrunedData`.
        """

        if get_zero_channels := BW_PRUNE.get(node.op_type):
            indices_of_inputs = get_zero_channels(self, node, pruned)
            stopper = False
        else:
            indices_of_inputs = [set()] * len(node.input)
            stopper = True
        for pred, ind in zip(self.graph.onnx_predecessors(node), indices_of_inputs):
            if pred not in pruned.bw_zero_indices:
                pruned.bw_zero_indices[pred] = ind
            else:
                # if two nodes go backward to a same output port
                # the available zero channels should be the intersection of both
                pruned.bw_zero_indices[pred] &= ind
            if stopper:
                pruned.bw_stopper[pred] = True
        logger.debug(f"zero in channels of {node.name}: {indices_of_inputs}")

    def update_pruned_axes(self, node: NodeProto, memo: PrunedData) -> bool:
        """Calculate the final pruned axes by mixing the results from upstream and
        sibling nodes.
        """
        is_output_node = self.graph.out_degree(node.name) == 0
        need_update = is_output_node and node in memo.fw_zero_indices
        need_update |= node in memo.bw_zero_indices and node in memo.fw_zero_indices
        need_refresh = False
        if need_update:
            # need a mix
            if mix_channels := MIXTURE.get(node.op_type):
                out_renew_axes = mix_channels(self, node, memo)
            else:
                out_renew_axes = _default_mixture(self, node, memo)
            indices = set() if is_output_node else out_renew_axes
            need_refresh = memo.fw_zero_indices[node] != indices
            memo.fw_zero_indices[node] = memo.axes_map[node] = indices
            logger.debug(f"pruned out channels of {node.name}: {indices}")
        return need_refresh

    def remove_weights_channels(self, node: NodeProto, memo: PrunedData):
        """Remove zero output channels in weights."""

        if node.name in memo.seen_nodes:
            return

        out_axes = memo.axes_map.get(node)
        producer = self.get_input_node(node, 0)
        if out_axes is None:
            logger.debug(f"axes map of {node.name} is empty")
            return
        if producer is None:
            in_axes: Set[int] | None = set()
        else:
            in_axes = memo.axes_map.get(producer)
        if in_axes is None:
            logger.debug(f"input axes of {node.name} is empty")
            return
        elif not in_axes and not out_axes:
            logger.debug(f"not a sparse op: {node.name}")
            memo.seen_nodes.add(node.name)
            return  # not a sparse op

        weights = _get_weights(self, node)
        out_channel_axis = 0
        in_channel_axis = 1
        if node.op_type in ("ConvTranspose",):
            out_channel_axis = 1
            in_channel_axis = 0
        kept0 = set(range(weights.shape[out_channel_axis])) - out_axes
        kept1 = set(range(weights.shape[in_channel_axis])) - in_axes
        pruned_weights = np.take(weights, tuple(kept0), axis=out_channel_axis)
        pruned_weights = np.take(pruned_weights, tuple(kept1), axis=in_channel_axis)
        weight_node = make_constant(f"{node.name}/pruned_weight", pruned_weights)
        self += weight_node
        node.input[1] = weight_node.output[0]

        if len(node.input) > 2:
            # update bias
            bias = self.get_value(node.input[2])
            if bias is None:
                raise ValueError(f"Cannot get a constant value of bias: {node.name}")
            pruned_bias = np.take(bias, tuple(kept0))
            bias_node = make_constant(f"{node.name}/pruned_bias", pruned_bias)
            self += bias_node
            node.input[2] = bias_node.output[0]
        self += node
        memo.seen_nodes.add(node.name)
        logger.debug(
            f"update weights of {node.name}: {weights.shape}->{pruned_weights.shape}"
        )
