# PyTorch custom operators

This domain register custom operator symbolic functions in PyTorch for those operators that are not supported by ONNX.

## Guidance
```python
@onnxscript.script(Opset(domain="pytorch.aten", version=2))
def diag(x, diagonal: int):
    """Implementation of ``aten::diag`` using ONNX operators."""

    shape = op.Shape(x)
    idv = op.ConstantOfShape(shape, value=from_array(np.array([1], np.float32)))
    idv_x = op.CastLike(idv, x)
    idv_t = op.Transpose(idv_x)
    y = op.MatMul(x, idv_t)
    mask = op.EyeLike(y, k=diagonal)
    return op.Mul(y, mask)


@onnx_symbolic("aten::diag", 1, custom=True)
@torch.onnx.symbolic_helper.parse_args("v", "i")
def diag_symbolic(g: GraphContext, x: torch._C.Value, diagonal: int = 0):
    """Symbolic function for ``torch.diag``."""

    return g.onnxscript_op(diag, x, diagonal_i=diagonal).setTypeAs(x)
```

The above code register a symbolic function for `aten::diag` which is `torch.diag` in PyTorch. `aten::diag` is the operator name of `torch.diag`, which can be viewed in the error log when trying to export it directly.

The implementation of the symbolic function is optional, the above code implements a `diag` using [onnxscript](https://onnxscript.ai). When an implementation is not possible, the symbolic function can return `g.op("diag", x, diagonal_i=diagonal).setTypeAs(x)` to indicate a dummy operator.

The `diagonal_i` declare an int attribute to the operator. Where the suffix can be "_i", "_f", "_s" or "_t", representing int/ints, float/floats, string or tensor respectively.

## Operation List

- [x] `aten::diag`
