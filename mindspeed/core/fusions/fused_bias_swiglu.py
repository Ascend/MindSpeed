import torch
import torch_npu


def _check_empty_weighted_swiglu(y, weights):
    # The native kernel does not accept empty tensors. Only the standard empty
    # expert has two empty gradients; e.g. [tokens, 0] needs a nonempty zero wgrad.
    if y.ndim != 2 or y.shape[0] != 0 or y.shape[1] % 2 or weights.shape != (0, 1):
        raise ValueError('Empty weighted SwiGLU requires input [0, 2H] and weights [0, 1].')


# Register these helpers directly, without jit_fuser: its NPU shim is named
# "wrapper", which the patch manager interprets as a patch decorator.
def weighted_swiglu(y, weights):
    """Fuse the activation while preserving Megatron's weighted output dtype."""
    if y.numel() == 0:
        _check_empty_weighted_swiglu(y, weights)
        return y.new_empty((0, y.shape[1] // 2))
    return (fused_swiglu(y.contiguous()) * weights).to(y.dtype)


def weighted_swiglu_back(g, y, weights):
    """Use the native SwiGLU derivative; keep router weighting in its own dtype."""
    if torch.is_grad_enabled():
        raise RuntimeError('Fused weighted SwiGLU supports first-order backward only.')
    if y.numel() == 0:
        _check_empty_weighted_swiglu(y, weights)
        return (
            torch.empty_like(y, memory_format=torch.contiguous_format),
            torch.empty_like(weights, memory_format=torch.contiguous_format),
        )

    weighted_grad = g * weights
    # SwiGluGrad requires matching dtypes. Keep FP32 router weighting rather
    # than rounding the weighted gradient down before differentiating.
    backward_input = y.to(weighted_grad.dtype).contiguous()
    input_grad = torch_npu.npu_swiglu_backward(weighted_grad.contiguous(), backward_input, dim=-1).to(y.dtype)
    del backward_input, weighted_grad
    # Recompute this activation instead of extending its lifetime across the
    # forward/backward boundary. The original Function still saves y and weights.
    activation = fused_swiglu(y.contiguous())
    weights_grad = torch.sum(activation * g.to(weights.dtype), dim=-1, keepdim=True).to(weights.dtype)
    return input_grad, weights_grad


def fused_swiglu(x):
    return torch_npu.npu_swiglu(x, dim=-1)


class SwiGLUFunction:
    @staticmethod
    def apply(x, *args):
        return fused_swiglu(x)


class BiasSwiGLUFunction:
    @staticmethod
    def apply(x, bias, *args):
        return fused_swiglu(x + bias)
