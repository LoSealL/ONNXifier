---
name: finding-existing-passes
description: Use when about to implement a new ONNX optimization pass, rewriter, or op-transformation (eliminate/fuse/swap/canonicalize ops) in this repo — before writing any new code. Also use when unsure which passes exist or how to activate one.
---

# Finding Existing Passes

## Overview

The pass registry is the authoritative, complete inventory of every optimization in this repo. Query it with `onnxify --print` BEFORE writing a new pass — a pass that handles the need usually already exists (often under a name you wouldn't grep for, e.g. `prelu_to_leaky`, `swap_concat`).

**Violating the letter of this skill (skipping the registry check) is violating its spirit: reusing an existing pass beats writing a duplicate.**

## Quick Reference

Run from the repo root:

| Command                                                | Shows                                                                   |
| ------------------------------------------------------ | ----------------------------------------------------------------------- |
| `uv run onnxify --print`                               | all registered passes (name, DEPS, PATCH, CONFIG)                       |
| `uv run onnxify --print l1` / `l2` / `l3`              | passes of one optimization level                                        |
| `uv run onnxify --print <pass_name>`                   | one specific pass                                                       |
| `uv run onnxify --print all --full`                    | + DOC column: the rewriter's docstring (its manual, like `help(class)`) |
| `uv run onnxify --print all --full --print-format csv` | grep-friendly CSV                                                       |
| `... --print-format json`                              | machine-readable JSON                                                   |

Search by concept (keyword, not exact name):

```powershell
uv run onnxify --print all --full --print-format csv | Select-String -Pattern "resize"
```

Column meanings:

- **DEPS**: passes run automatically before it — you get these for free when activating it
- **PATCH**: passes run automatically after it
- **CONFIG**: the pass's signature — pass-specific options go on the CLI as `--key value` after the input model, or in the `-c` config file

## Workflow

1. `--print all --full --print-format csv` and search the DOC column for your concept (try several synonyms: transpose/permute, fold/constant, eliminate/remove, swap/replace).
2. Found a match? Read that one file under `onnxifier/passes/<category>/<name>.py` and its test under `tests/passes/`.
3. Only if nothing matches (check synonyms again): write a new pass in the matching category folder — `auto_load` registers it via `@PASSES.register(...)` / `@L1|L2|L3.register(...)`.

## Using a Found Pass

```powershell
uv run onnxify model.onnx -a resize_to_convtransposed --checker-backend onnx
```

`-a` (`--activate`) selects passes by registry name (comma-separated); `-r` removes. Without `-a`, L1+L2+L3 run.

## Common Mistakes

| Mistake                                       | Fix                                                                                        |
| --------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Grepping source for op names to find a pass   | Registry lookup first — catches passes whose names/docstrings don't contain your grep term |
| Writing a new pass for a need already covered | Extend the existing pass + its test file instead of adding a new file                      |
| Guessing CLI flags                            | It's `-a/--activate` (not `-p`), `-r/--remove`, `-c/--config-file`                         |
| Passing pass options in the wrong place       | They go AFTER the input model as `--key value`, or in the `-c` JSON file                   |
