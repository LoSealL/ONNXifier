# ONNXifier Repository Description

## Purpose

ONNXifier converts supported model IRs into ONNX and can optionally optimize the result
through a registry-driven pass pipeline.

## Core Layout

- onnxifier/__main__.py
  CLI entrypoint for conversion, pass selection, config loading, and validation flags.
- onnxifier/pass_manager.py
  Expands pass dependencies and patch passes, then applies them in order to an OnnxGraph.
- onnxifier/graph.py
  Main ONNX graph wrapper used by passes and conversion logic.
- onnxifier/passes/
  Optimization and rewrite passes, grouped by categories such as canonicalization,
  fusion, quantize, swap, version_converter, and experiments.
- onnxifier/passes/__init__.py
  Defines the Registry objects PASSES, L1, L2, and L3 and auto-loads pass folders.
- onnxifier/passes/rewriter.py
  Base Rewriter class. Pattern-driven class passes typically inherit from this.
- onnxifier/passes/pattern.py
  Pattern matching abstractions used by rewrites.
- onnxifier/evaluator/
  Runtime evaluation helpers used by passes such as constant folding.
- onnxifier/domain/
  Source IR frontends and domain-specific conversion logic.
- tests/
  Repository tests. Pass-specific regressions usually live under tests/passes/.

## Pass Authoring Conventions

### Registration

- Class-based graph rewrite passes usually use decorators such as @L1.register(...),
  @L2.register(...), or @L3.register(...).
- Function-style passes can also be registered and are invoked directly by PassManager.
- deps run before the pass. patch runs after the pass.

### Class Passes

- Inherit from Rewriter.
- Provide a Pattern in __init__.
- Implement rewrite(self, graph, nodes, ...).
- Mutate graphs through Rewriter helpers like self += nodes_to_add and self -= nodes.

### Function Passes

- Useful for simpler whole-graph transforms.
- PassManager treats plain functions differently from class-based rewrites when node
  filtering is requested.

## Test Strategy

- Prefer focused tests that build tiny ONNX graphs with onnx.helper utilities.
- Put pass regressions in tests/passes/test_<pass_name>.py when the change is specific
  to one optimization pass.
- Use tests/test_pass_manager.py for dependency ordering, recursive execution, config
  wiring, and node-selection behavior.
- Keep assertions behavioral: node count, operator presence, attribute values, graph
  connectivity, or outputs after evaluation.

## Common Commands

- Install base package: `uv sync`
- Install test dependencies: `uv sync --dev`
- Run one pass test: `uv run --dev pytest tests/passes/test_fold_constant.py -q -s`
- Run all pass tests: `uv run --dev pytest tests/passes -q -s`
- Type check: `uv run --group dev pyright onnxifier`
- Ruff style check: `uv run --group dev ruff check onnxifier tests`

## Practical How-To

### Query ONNX Operator Docs

- ONNX operator docs URL pattern:
  `https://onnx.ai/onnx/operators/onnx__<OpName>.html`
- Example: `https://onnx.ai/onnx/operators/onnx__MatMul.html`
- Validate pass assumptions against official operator contracts (inputs, outputs,
  attributes, shape inference notes) before implementing rewrites.

### Pass Folder Selection Guide

- `onnxifier/passes/fusion/`: fuse multi-op patterns into fewer ops.
- `onnxifier/passes/fission/`: decompose a complex op into simpler ops.
- `onnxifier/passes/swap/`: replace equivalent implementation patterns.
- `onnxifier/passes/canonicalization/`: normalize representation to canonical form.
- `onnxifier/passes/quantize/`: quantization-related transforms.
- `onnxifier/passes/version_converter/`: opset/version migration transforms.
- `onnxifier/passes/globals/`: whole-graph cleanup and global transforms.
- `onnxifier/passes/experiments/`: experimental, not-yet-stable passes.

Rule of thumb: choose the folder by primary transformation intent, not by incidental
implementation details.

## Editing Heuristics

- Match the existing file's current style and typing approach.
- Avoid broad refactors when fixing a pass regression.
- If a pass relies on evaluator semantics, confirm whether the intended behavior should
  live in the pass or in shared evaluator/graph utilities.
- When adding a new pass, place it in the most specific category folder and add a
  dedicated regression test.
