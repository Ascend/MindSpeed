from functools import lru_cache

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


@lru_cache
def get_quant_dtype():
    args = get_args()
    if args.fp8 == 'hif8':
        return QuantDtype(torch_npu.hifloat8, torch_npu.hifloat8, torch_npu.hifloat8)
    elif args.fp8 == 'hybrid':
        return QuantDtype(torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e5m2)
    return QuantDtype(torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e4m3fn)


def get_hccl_comm_name(group, rank):
    if torch.__version__ > "2.0":
        global_rank = torch.distributed.get_global_rank(group, rank)
        hcomm_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
    else:
        hcomm_name = group.get_hccl_comm_name(rank)

    return hcomm_name


def all_gather_along_dim(input_, async_op=False, axis=0):
    from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
    group = get_tensor_model_parallel_group()
    world_size = get_tensor_model_parallel_world_size()
    dim_size = list(input_.size())
    dim_size[axis] = dim_size[axis] * world_size
    output_ = torch.empty(dim_size, dtype=input_.dtype, device=torch.npu.current_device(), requires_grad=False)
    handle = torch.distributed._all_gather_base(output_, input_.contiguous(), group=group, async_op=async_op)
    return handle, output_
