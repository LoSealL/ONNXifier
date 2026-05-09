# ONNXifier

[English](README.md) | [中文](README_CN.md)

A simple tool to convert any IR format to ONNX file.

[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

| Framework | Status |
|:----------|:-------|
| OpenVINO    | ✅  |
| ONNXRuntime | ✅  |
| TensorRT    | 🚧  |
| TensorRT-LLM | 🚧 |

- ✅: well supported
- 🪛: partially supported
- 🚧: developing

## Usage

1. Install from PyPI
```shell
pip install onnxifier
```

2. Convert IR using CLI
```shell
onnxify model.xml
```

```
usage: onnxify input_model.onnx [output_model.onnx]

onnxify command-line api

options:
  -h, --help            show this help message and exit
  --install-completion [{bash,pwsh}]
                        install shell completion for the specified shell and
                        exit.
  -a [ACTIVATE ...], --activate [ACTIVATE ...]
                        select passes to be activated, activate L1, L2 and L3
                        passes if not set.
  -r [REMOVE ...], --remove [REMOVE ...]
                        specify passes to be removed from activated passes.
  -n, --no-passes       do not run any optimizing passes, just convert the
                        model
  --print [PRINT]       print the name of all optimizing passes
  --format {protobuf,textproto,json,onnxtxt}
                        onnx file format
  -s, --infer-shapes    infer model shapes
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        specify a json-format config file for passes
  -u, --uncheck         no checking output model
  --check               check optimized model with random inputs
  -d, --dry-run         only run passes without saving the output model
  --checker-backend {onnx,openvino,onnxruntime}
                        backend for accuracy checking, defaults to onnxruntime
  -v OPSET_VERSION, --opset-version OPSET_VERSION
                        target opset version, defaults to 20
  -vv [{DEBUG,INFO,WARNING,ERROR,CRITICAL}], --log-level [{DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                        specify the level of log messages to be printed,
                        defaults to INFO
  -R, --recursive       recursively optimize nested functions
  --nodes [NODES ...]   specify a set of node names to apply passes only on
                        these nodes
```

To print pass information:

```shell
onnxify --print all
onnxify --print fuse_swish
onnxify --print l1
```

### Shell Completion

`onnxify` provides tab-completion for **Bash** and **PowerShell** to help you quickly select pass names and options.

After installing `onnxifier` from PyPI, run the built-in installer once for your shell:

**Bash**
```shell
onnxify --install-completion bash
source ~/.bashrc
```

**PowerShell**
```powershell
onnxify --install-completion pwsh
. $PROFILE
```

The installer is idempotent — running it again will not duplicate entries in your profile.

Once enabled, you can use `Tab` to complete pass names after `-a` / `-r` / `--print`, for example:

```shell
# Complete a single pass
onnxify model.onnx -a ins<TAB>
# → inspect_sparsity_ratio inspect_weights_distribution insert_conv_before_act_shave

# Complete multiple space-separated passes
onnxify model.onnx -a infer_shape fold_const<TAB>
# → fold_constant

# Complete comma-separated passes
onnxify model.onnx -a fuse_gelu,ins<TAB>
# → fuse_gelu,inspect_sparsity_ratio ...

# Complete --print arguments
onnxify --print l<TAB>
# → l1 l2 l3
```

## Custom Domain Shape Inference

ONNXifier supports shape inference for custom domain ops (e.g., `trt::CausalConv1d`, `com.microsoft::MyOp`) through a registration API.

### Usage

Shape inference for domain ops is automatic when using the `--infer-shapes` flag:

```shell
onnxify model_with_trt_ops.onnx --infer-shapes
```

### Registering Shape Inference for Custom Ops

Developers register shape inference using **ONNXScript** functions. The decorator inserts the function into the model during `infer_shapes`, then cleans it up afterward.

```python
import onnxscript
from onnxscript.onnx_opset import opset19 as op
from onnxscript.values import Opset

from onnxifier.domain.shape_inference import register_shape_inference

@register_shape_inference("com.mycompany", "MyOp")
@onnxscript.script(Opset("com.mycompany", 1), default_opset=op)
def my_op_shape_infer(input_0, input_1):
    # Return shapes for each output
    return op.Identity(input_0), op.Identity(input_1)
```

If domain/op_type are omitted, they are inferred from the ONNXScript function metadata:

```python
@register_shape_inference()  # Uses function.name and function.opset.domain
@onnxscript.script(Opset("com.mycompany", 1), default_opset=op)
def MyOp(input_0):
    return op.Identity(input_0)
```

See [quickstart.md](specs/001-custom-domain-shape-inference/quickstart.md) for detailed examples.

## TODO

- [ ] [**OV**] Add [Loop](https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/loop-5.html) support.
- [ ] [**OV**] Add [NMS](https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/sort/non-max-suppression-9.html) support.
- [ ] [**OV**] Add [If](https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/condition/if-8.html) support.
- [ ] [**ONNX**] Support to optimize [If](https://onnx.ai/onnx/operators/onnx__If.html).


## Contribute

1. pyright type checking

```
pip install -U pyright
pyright onnxifier
```

2. mypy type checking

```
pip install -U mypy
mypy onnxifier --disable-error-code=import-untyped --disable-error=override --disable-error=call-overload
```

3. pre-commit checking

```
pip install -U pre-commit
pre-commit run --all-files
```
