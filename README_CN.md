# ONNXifier

[English](README.md) | [中文](README_CN.md)

一个将任意 IR 格式转换为 ONNX 文件的简单工具。

[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

| 框架        | 状态   |
|:------------|:-------|
| OpenVINO    | ✅     |
| ONNXRuntime | ✅     |
| TensorRT    | 🚧     |
| TensorRT-LLM | 🚧    |

- ✅: 完善支持
- 🪛: 部分支持
- 🚧: 开发中

## 使用方法

1. 从 PyPI 安装
```shell
pip install onnxifier
```

2. 使用 CLI 转换 IR
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

打印 Pass 信息：

```shell
onnxify --print all
onnxify --print fuse_swish
onnxify --print l1
```

### Shell 自动补全

`onnxify` 为 **Bash** 和 **PowerShell** 提供了 Tab 自动补全功能，帮助你快速选择 Pass 名称和选项。

从 PyPI 安装 `onnxifier` 后，针对你的 Shell 运行一次内置安装器即可：

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

安装器具有幂等性 —— 重复运行不会在你的配置文件中产生重复条目。

启用后，你可以在 `-a` / `-r` / `--print` 后按 `Tab` 补全 Pass 名称，例如：

```shell
# 补全单个 Pass
onnxify model.onnx -a ins<TAB>
# → inspect_sparsity_ratio inspect_weights_distribution insert_conv_before_act_shave

# 补全多个空格分隔的 Pass
onnxify model.onnx -a infer_shape fold_const<TAB>
# → fold_constant

# 补全逗号分隔的 Pass
onnxify model.onnx -a fuse_gelu,ins<TAB>
# → fuse_gelu,inspect_sparsity_ratio ...

# 补全 --print 参数
onnxify --print l<TAB>
# → l1 l2 l3
```

## 自定义算子形状推导

开发者使用 **ONNXScript** 函数注册自定义域算子的形状推导。装饰器会在 `infer_shapes` 期间将函数临时插入模型，推导完成后自动清理。

```python
import onnxscript
from onnxscript.onnx_opset import opset19 as op
from onnxscript.values import Opset

from onnxifier.domain.shape_inference import register_shape_inference

@register_shape_inference("com.mycompany", "MyOp")
@onnxscript.script(Opset("com.mycompany", 1), default_opset=op)
def my_op_shape_infer(input_0, input_1):
    # 为每个输出返回对应的形状
    return op.Identity(input_0), op.Identity(input_1)
```

如果省略 domain/op_type，则从 ONNXScript 函数元数据自动推断：

```python
@register_shape_inference()  # 使用 function.name 和 function.opset.domain
@onnxscript.script(Opset("com.mycompany", 1), default_opset=op)
def MyOp(input_0):
    return op.Identity(input_0)
```

详细示例请参阅 [quickstart.md](specs/001-custom-domain-shape-inference/quickstart.md)。

## TODO

- [ ] [**OV**] 添加 [Loop](https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/loop-5.html) 支持。
- [ ] [**OV**] 添加 [NMS](https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/sort/non-max-suppression-9.html) 支持。
- [ ] [**OV**] 添加 [If](https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/condition/if-8.html) 支持。
- [ ] [**ONNX**] 支持优化 [If](https://onnx.ai/onnx/operators/onnx__If.html)。


## 贡献代码

1. pyright 类型检查

```
pip install -U pyright
pyright onnxifier
```

2. mypy 类型检查

```
pip install -U mypy
mypy onnxifier --disable-error-code=import-untyped --disable-error=override --disable-error=call-overload
```

3. pre-commit 检查

```
pip install -U pre-commit
pre-commit run --all-files
```
