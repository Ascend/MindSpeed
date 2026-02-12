from .float8_tensor import Float8Tensor, Float8Tensor2D
from .float8_tensor_cpu import Float8TensorCpu
from .mxfp8_tensor import MXFP8Tensor
from .mxfp8_tensor_cpu import MXFP8TensorCpu
from .float8_block_tensor import Float8BlockTensor

FP8_TENSOR = (Float8Tensor, Float8Tensor2D)


def is_fp8_tensor(tensor):
    return isinstance(tensor, FP8_TENSOR)


def is_fp8_tensor_2d(tensor):
    return isinstance(tensor, Float8Tensor2D)
