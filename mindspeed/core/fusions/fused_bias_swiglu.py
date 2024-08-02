import torch_npu


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
