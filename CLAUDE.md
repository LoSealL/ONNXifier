# ONNXifier — Principles for Claude

## Project Context

ONNXifier is a Python CLI tool that converts IR formats (OpenVINO, TensorRT, etc.) to ONNX and applies optimization passes. It is library-first: all logic lives in the `onnxifier` package, exposed through the `onnxify` CLI.

## Non-Negotiable Rules

### 1. ONNX Correctness

- Every output model MUST pass the official ONNX checker
- Conversions MUST preserve semantics; optimizations MUST NOT change numerical results beyond documented tolerance
- Always validate with randomized inputs against a reference backend

### 2. CLI-First Design

- All features MUST be accessible via `onnxify` CLI
- Python API is secondary and mirrors CLI capabilities
- Use typer for CLI implementation

### 3. Pass Modularity

- Each optimization is an isolated, composable pass
- Passes have single responsibility and explicit dependencies
- Passes are registered in the pass manager with L1/L2/L3 levels

### 4. Test-First

- Write tests before or concurrently with pass implementation
- Required: unit tests, integration tests, edge cases, accuracy comparisons
- Tests MUST fail before implementation is complete

### 5. Type Safety

- All code MUST pass pyright strict checking
- All public APIs MUST have complete type annotations
- Run `pyright onnxifier` and `mypy` before committing

### 6. Coding Style

- All code MUST NOT violate ruff check and ruff format
- Run `ruff check onnxifier --unsafe-fixes --fix` and `ruff format onnxifier`

## Workflow

- Run in virtualenv via `uv`: `uv run onnxify ...`, `uv run python ...`
- If current git workspace is clean, checkout to a new branch and commit after task finished.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan at:
specs/001-custom-domain-shape-inference/plan.md
<!-- SPECKIT END -->
