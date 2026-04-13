---
name: onnxifier-dev
description: 'Use when working on ONNXifier internals: ONNX graph rewrites, optimization passes, PassManager behavior, pass registration, CLI conversion flow, domain adapters, and pytest coverage for graph or pass changes. Triggers include add pass, debug pass, update ONNX transformation, fix failing pass test, inspect Rewriter or Pattern usage, and explain ONNXifier architecture.'
argument-hint: 'Describe the ONNXifier task, affected pass or module, and expected behavior'
user-invocable: true
---

# ONNXifier Development

This skill is for repository-specific work in ONNXifier. Load it when the task involves
adding or changing optimization passes, debugging graph rewrites, tracing how the CLI
and PassManager invoke passes, or updating tests around ONNX graph behavior.

## When To Use

- Add or modify a pass under onnxifier/passes/
- Debug a pass dependency, patch pass, or PassManager execution order
- Update ONNX graph rewrites implemented with Rewriter and Pattern
- Change CLI behavior in onnxifier/__main__.py
- Add or fix tests under tests/passes/ or tests/
- Explain repository architecture before making changes

## Working Rules

1. Start from the smallest relevant surface area.
   Read the target pass, its neighboring passes in the same category, and the matching
   test file before editing.
2. Prefer existing abstractions.
   Reuse OnnxGraph, Rewriter, Pattern, Evaluator, and helper utilities instead of
   introducing one-off graph manipulation code.
3. Keep pass behavior local and testable.
   Most changes should come with a focused test in tests/passes/ or a nearby unit test.
4. Preserve registry conventions.
   Passes are typically registered through L1, L2, L3, or PASSES and may declare deps
   and patch passes.
5. Verify at the narrowest useful scope first.
   Run the smallest relevant pytest target before broader validation.

## Standard Workflow

1. Identify the change surface.
   Check whether the task belongs to graph core, pass implementation, domain import,
   evaluator logic, or CLI wiring.
2. Inspect the implementation path.
   For pass work, read the pass file, onnxifier/passes/__init__.py, onnxifier/rewriter
   support code, and the closest tests.
3. Implement minimal changes.
   Match the repository's current style, typing level, and naming patterns.
4. Add or update tests.
   Prefer precise regression tests that construct tiny ONNX graphs in code.
5. Validate.
   Run the targeted pytest file first, then a broader test slice only if needed.

## Validation Shortcuts

- Single pass test file: pytest tests/passes/test_<pass_name>.py -q -s
- Core unit test: pytest tests/test_pass_manager.py -q -s
- Broader pass suite: pytest tests/passes -q -s
- Type checking if relevant: pyright onnxifier

## Practical Guides

### Query ONNX Op Documentation

- Use the ONNX operator docs URL pattern:
   `https://onnx.ai/onnx/operators/onnx__<OpName>.html`
- Example: `If` -> `https://onnx.ai/onnx/operators/onnx__If.html`
- For non-ONNX source IR semantics (for example OpenVINO ops), consult domain docs,
   then map behavior to ONNX operator definitions before implementing passes.

### Coding Style Check (Ruff)

- Run style checks with project dependency groups:
   `uv run --group dev ruff check onnxifier tests`
- Auto-fix safe style issues when needed:
   `uv run --group dev ruff check onnxifier tests --fix`
- Current Ruff selection comes from `pyproject.toml`:
   `E`, `F`, `UP`, `I` with line length `88`.

### Decide Which Pass Folder To Use

- Place the pass in the most specific category under `onnxifier/passes/`.
- Use this decision guide:
   - `fusion/`: multiple ops -> fewer ops (pattern fusion)
   - `fission/`: one op -> multiple ops (decomposition)
   - `swap/`: equivalent replacement pattern (A -> B)
   - `canonicalization/`: normalize attributes/types/forms to canonical ONNX style
   - `quantize/`: quantization/dequantization related transforms
   - `version_converter/`: opset/version compatibility transforms
   - `globals/`: whole-graph cleanup/analysis passes
   - `experiments/`: unstable or exploratory passes
- If uncertain between two folders, choose by dominant intent (semantic transform type),
   then add a focused regression test under `tests/passes/`.

## Repository Notes

See [repo description](./references/repo-description.md) for:

- module map and ownership boundaries
- pass authoring conventions
- test placement rules
- common commands and editing heuristics
