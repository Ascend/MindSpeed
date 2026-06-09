import importlib

import torch

__all__ = [
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
]

_ORIGINAL_TORCH_SUM = torch.sum
_ORIGINAL_TENSOR_SUM = torch.Tensor.sum

_batch_invariant_MODE = False
_batch_invariant_LIB = None


def _origin_reduce_sum(x, dim=None, keepdim=False, dtype=None, out=None):
    kwargs = {"dim": dim, "keepdim": keepdim, "dtype": dtype}
    if out is not None:
        kwargs["out"] = out
    return _ORIGINAL_TORCH_SUM(x, **kwargs)


def _sum_to_shape(grad: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    while grad.dim() > len(shape):
        grad = grad.sum(dim=0)
    for i, size in enumerate(shape):
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(dim=i, keepdim=True)
    return grad.reshape(shape)


def _is_single_reduce_dim(dim) -> bool:
    return not isinstance(dim, (tuple, list)) or len(dim) == 1


def _is_sum_dtype_supported(dtype) -> bool:
    return dtype not in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool]


def _should_use_origin_reduce_sum(x, dim, out) -> bool:
    if dim is None or out is not None:
        return True
    if not _is_single_reduce_dim(dim):
        return True
    if x.device.type != "npu":
        return True
    return not _is_sum_dtype_supported(x.dtype)


class MmBatchInvariantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return torch.ops.batch_invariant_ops.npu_mm_batch_invariant(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = torch.matmul(grad_output, b.transpose(-2, -1)) if ctx.needs_input_grad[0] else None
        grad_b = torch.matmul(a.transpose(-2, -1), grad_output) if ctx.needs_input_grad[1] else None
        return grad_a, grad_b


def mm_adapter(a, b):
    return MmBatchInvariantFunction.apply(a, b)


class MatmulBatchInvariantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        ctx.a_dim = a.dim()
        ctx.b_dim = b.dim()
        ctx.save_for_backward(a, b)
        return torch.ops.batch_invariant_ops.npu_matmul_batch_invariant(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        a_mat = a.unsqueeze(0) if ctx.a_dim == 1 else a
        b_mat = b.unsqueeze(-1) if ctx.b_dim == 1 else b

        if ctx.a_dim == 1 and ctx.b_dim == 1:
            grad_a = grad_output * b if ctx.needs_input_grad[0] else None
            grad_b = grad_output * a if ctx.needs_input_grad[1] else None
            return grad_a, grad_b

        if ctx.a_dim == 1:
            grad_mat = grad_output.unsqueeze(-2)
        elif ctx.b_dim == 1:
            grad_mat = grad_output.unsqueeze(-1)
        else:
            grad_mat = grad_output

        grad_a = None
        grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = torch.matmul(grad_mat, b_mat.transpose(-2, -1))
            if ctx.a_dim == 1:
                grad_a = grad_a.squeeze(-2)
            grad_a = _sum_to_shape(grad_a, ctx.a_shape)
        if ctx.needs_input_grad[1]:
            grad_b = torch.matmul(a_mat.transpose(-2, -1), grad_mat)
            if ctx.b_dim == 1:
                grad_b = grad_b.squeeze(-1)
            grad_b = _sum_to_shape(grad_b, ctx.b_shape)
        return grad_a, grad_b


def matmul_adapter(a, b):
    return MatmulBatchInvariantFunction.apply(a, b)


class ReduceSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, keepdim, dtype):
        ctx.save_for_backward(x)
        ctx.keepdim = keepdim
        ctx.original_shape = x.shape

        if dtype is not None and dtype != x.dtype:
            x = x.to(dtype)

        if isinstance(dim, (tuple, list)):
            dim = dim[0]

        ndim = x.dim()
        if dim < 0:
            dim = ndim + dim
        ctx.dim = dim

        if dim == ndim - 1:
            result = torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(x, dim, keepdim)
        else:
            perm = list(range(ndim))
            perm.remove(dim)
            perm.append(dim)
            x_permuted = x.permute(perm)
            result = torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant(
                x_permuted, -1, keepdim
            )
            if keepdim:
                reverse_perm = [0] * len(perm)
                for i, p in enumerate(perm):
                    reverse_perm[p] = i
                result = result.permute(reverse_perm)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        dim = ctx.dim
        keepdim = ctx.keepdim
        original_shape = ctx.original_shape

        if dim is None:
            return grad_output.expand_as(input_tensor), None, None, None

        if dim < 0:
            dim = len(original_shape) + dim

        if not keepdim:
            grad_output = grad_output.unsqueeze(dim)

        return grad_output.expand_as(input_tensor), None, None, None


def reduce_sum_adapter(
    x: torch.Tensor, dim=None, keepdim: bool = False, dtype=None, axis=None, out=None
):
    if axis is not None and dim is not None:
        raise ValueError("Cannot specify both 'dim' and 'axis'. Use only 'dim'.")
    if axis is not None:
        dim = axis
    if _should_use_origin_reduce_sum(x, dim, out):
        return _origin_reduce_sum(x, dim=dim, keepdim=keepdim, dtype=dtype, out=out)
    return ReduceSumFunction.apply(x, dim, keepdim, dtype)


class LogSoftmaxBatchInvariant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        y = torch.ops.batch_invariant_ops.npu_log_softmax_batch_invariant(x, dim)
        ctx.save_for_backward(y)
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        dim = ctx.dim
        sum_grad = grad_output.sum(dim=dim, keepdim=True)
        return grad_output - torch.exp(y) * sum_grad, None


def log_softmax_adapter(x, dim, dtype=None):
    if dtype is not None and dtype != x.dtype:
        x = x.to(dtype)
    return LogSoftmaxBatchInvariant.apply(x, dim)


def _log_softmax_adapter(x, dim, half_to_float=False):
    if half_to_float:
        x = x.float()
    return LogSoftmaxBatchInvariant.apply(x, dim)


def is_batch_invariant_mode_enabled():
    """Return True if global batch-invariant mode is currently enabled."""
    return _batch_invariant_MODE


def enable_batch_invariant_mode():
    """Enable global batch-invariant mode with NPU Ascend batch-invariant kernels."""
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_MODE:
        return
    importlib.import_module("batch_invariant_ops")
    importlib.import_module("torch_npu")

    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    _batch_invariant_LIB.impl("aten::mm", mm_adapter, "NPU")
    _batch_invariant_LIB.impl("aten::matmul", matmul_adapter, "NPU")
    _batch_invariant_LIB.impl("aten::sum", reduce_sum_adapter, "NPU")
    _batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_adapter, "NPU")
    _batch_invariant_LIB.impl("aten::log_softmax", log_softmax_adapter, "NPU")

    torch.sum = reduce_sum_adapter
    torch.Tensor.sum = reduce_sum_adapter


def disable_batch_invariant_mode():
    """Disable global batch-invariant mode and restore patched Python entry points."""
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None
    torch.sum = _ORIGINAL_TORCH_SUM
    torch.Tensor.sum = _ORIGINAL_TENSOR_SUM
