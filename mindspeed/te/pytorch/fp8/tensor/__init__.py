from .float8_tensor import Float8Tensor, Float8TensorWithTranspose
from .float8_tensor_cpu import Float8TensorCpu
from .mxfp8_tensor import MXFP8Tensor
from .mxfp8_tensor_cpu import MXFP8TensorCpu
from .groupwise_tensor import GroupwiseTensor


def is_fp8_tensor(tensor):
    return isinstance(tensor, Float8Tensor)


def is_fp8_tensor_with_trans(tensor):
    return isinstance(tensor, Float8TensorWithTranspose)