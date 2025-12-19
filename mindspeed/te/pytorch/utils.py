import torch
import torch_npu

from mindspeed.args_utils import get_full_args as get_args


def view_as_n_dim(input_tensor, dim=2):
    if dim < 2:
        raise AssertionError("dim should be greater than or equal to 2")
    if len(input_tensor.shape) != dim:
        return input_tensor.view(-1, *input_tensor.shape[-dim + 1:])
    return input_tensor


class QuantDtype:

    def __init__(self, x: torch.dtype, w: torch.dtype, grads: torch.dtype):
        self.x = x
        self.w = w
        self.grads = grads
        if self.x == torch_npu.hifloat8:
            self.mm_kwargs = {'x1_dtype': self.x, 'x2_dtype': self.w}
            self.gmm_kwargs = {"x_dtype": self.x, "weight_dtype": self.w}
        else:
            self.mm_kwargs = {}
            self.gmm_kwargs = {}


qdtype = None


def get_quant_dtype():
    global qdtype
    if qdtype:
        return qdtype
    args = get_args()
    if args.fp8 == 'hif8':
        qdtype = QuantDtype(torch_npu.hifloat8, torch_npu.hifloat8, torch_npu.hifloat8)
    elif args.fp8 == 'hybrid':
        qdtype = QuantDtype(torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e5m2)
    qdtype = QuantDtype(torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e4m3fn)
    return qdtype
