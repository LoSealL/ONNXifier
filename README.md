# ONNXifier
A simple tool to convert any IR format to ONNX file.

[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

| Framework | Status |
|:----------|:-------|
| OpenVINO  |  ✅    |
| ONNXRuntime | ✅  |

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
usage: onnxify input_model.xml [output_model.onnx]

onnxify command-line api

options:
  -h, --help            show this help message and exit
  -a [ACTIVATE ...], --activate [ACTIVATE ...]
                        select passes to be activated, activate L1, L2 and L3 passes if not set.
  -r [REMOVE ...], --remove [REMOVE ...]
                        specify passes to be removed from activated passes.
  -n, --no-passes       do not run any optimizing passes, just convert the model
  --print [PRINT]       print the name of all optimizing passes
  --format {protobuf,textproto,json,onnxtxt}
                        onnx file format
  -s, --infer-shapes    infer model shapes
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        specify a json-format config file for passes
  -u, --uncheck         no checking output model
  --check               check optimized model with random inputs
  --checker-backend {onnx,openvino,onnxruntime}
                        backend for accuracy checking, defaults to openvino
  -v OPSET_VERSION, --opset-version OPSET_VERSION
                        target opset version, defaults to 19
  -vv [{DEBUG,INFO,WARNING,ERROR,CRITICAL}], --log-level [{DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                        specify the level of log messages to be printed, defaults to INFO
  -R, --recursive       recursively optimize nested functions
  --nodes [NODES ...]   specify a set of node names to apply passes only on these nodes
```

To print pass information:

```shell
onnxify --print all
onnxify --print fuse_swish
onnxify --print l1
```


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
