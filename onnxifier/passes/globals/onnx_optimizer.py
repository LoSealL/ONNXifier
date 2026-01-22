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

from contextlib import suppress
from typing import List, Optional

from ... import OnnxGraph
from .. import L3

with suppress(ImportError):
    import onnxoptimizer

    @L3.register()
    def onnx_optimizer(graph: OnnxGraph, passes: Optional[List[str]] = None):
        """Fuse op and remove isolated nodes.

        Args:
            graph (OnnxGraph): The onnx graph to be optimized.
            passes (List[str], optional): The optimization passes to be applied.
                Defaults to None.
        """

        if not passes:
            passes = [
                "eliminate_nop_cast",
                "eliminate_nop_dropout",
                "eliminate_nop_flatten",
                "eliminate_if_with_const_cond",
                "eliminate_nop_monotone_argmax",
                "eliminate_nop_pad",
                "eliminate_nop_concat",
                "eliminate_nop_split",
                "eliminate_nop_expand",
                "eliminate_shape_gather",
                "eliminate_slice_after_shape",
                "eliminate_nop_reshape",
                "eliminate_nop_with_unit",
                "eliminate_common_subexpression",
                "eliminate_deadend",
                "eliminate_identity",
                "eliminate_shape_op",
                "eliminate_unused_initializer",
                "eliminate_duplicate_initializer",
                "fuse_add_bias_into_conv",
                "fuse_bn_into_conv",
                "fuse_concat_into_reshape",
                "fuse_consecutive_concats",
                "fuse_consecutive_log_softmax",
                "fuse_consecutive_reduce_unsqueeze",
                "fuse_consecutive_slices",
                "fuse_consecutive_squeezes",
                "fuse_consecutive_transposes",
                "fuse_consecutive_unsqueezes",
                "fuse_matmul_add_bias_into_gemm",
                "fuse_pad_into_conv",
                "fuse_pad_into_pool",
                "fuse_qkv",
                "fuse_transpose_into_gemm",
                "extract_constant_to_initializer",
            ]
        # FIXME: onnxoptimizer ignores functions
        opt_model = onnxoptimizer.optimize(graph.model, passes)
        opt_model.functions.extend(graph.functions.values())
        return OnnxGraph(opt_model, base_dir=graph.external_base)
