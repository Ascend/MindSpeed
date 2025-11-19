import torch

from mindspeed.te.pytorch.fp8 import fp8_matmul, Float8Tensor, MXFP8Tensor
from mindspeed.te.pytorch.fp8.tensor import is_fp8_tensor
from mindspeed.te.pytorch.module.ops.comm_overlap_ops import CommOverlapOps, COMM_OVERLAP_CONFIG


class DefaultOps(CommOverlapOps):

    @staticmethod
    def allgather_matmul(input_, weight, bias, fp8_meta=None, key=None, fp8_enable=False):
        tp_world_size = COMM_OVERLAP_CONFIG.get_tp_size()
        tp_group = COMM_OVERLAP_CONFIG.get_tp_group()

        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * tp_world_size

        total_input = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        torch.distributed._all_gather_base(total_input, input_.contiguous(), group=tp_group, async_op=False)

        if not fp8_enable:
            output = torch.matmul(total_input, weight)
            return output, total_input, None
        else:
            if not is_fp8_tensor(total_input):
                total_input = fp8_meta.pre_compute(key[0], total_input)
            if not is_fp8_tensor(weight):
                weight = fp8_meta.pre_compute(key[1], weight)
            output = fp8_matmul(total_input, weight, fp8_meta, key)
            return output, total_input, weight

    @staticmethod
    def matmul_reduce_scatter(input_, weight, bias, fp8_meta=None, key=None, fp8_enable=False):
        tp_world_size = COMM_OVERLAP_CONFIG.get_tp_size()
        tp_group = COMM_OVERLAP_CONFIG.get_tp_group()

        if not fp8_enable:
            output_ = torch.matmul(input_, weight.t())
        else:
            if not is_fp8_tensor(input_):
                input_ = fp8_meta.pre_compute(key[0], input_)
            if not is_fp8_tensor(weight):
                weight = fp8_meta.pre_compute(key[1], weight)
            output_ = fp8_matmul(input_, weight, fp8_meta, key, (False, True))

        dim_size = list(output_.size())
        dim_size[0] = dim_size[0] // tp_world_size
        output = torch.empty(dim_size, dtype=output_.dtype, device=torch.cuda.current_device())

        torch.distributed._reduce_scatter_base(output, output_.contiguous(), group=tp_group)
        return output, input_, weight

    @staticmethod
    def matmul_all_reduce(input_, weight, bias, fp8_meta=None, key=None, fp8_enable=False):
        tp_world_size = COMM_OVERLAP_CONFIG.get_tp_size()
        tp_group = COMM_OVERLAP_CONFIG.get_tp_group()

        if not fp8_enable:
            output_ = torch.matmul(input_, weight.t())
        else:
            if not is_fp8_tensor(input_):
                input_ = fp8_meta.pre_compute(key[0], input_)
            if not is_fp8_tensor(weight):
                weight = fp8_meta.pre_compute(key[1], weight)
            output_ = fp8_matmul(input_, weight, fp8_meta, key, (False, True))

        if tp_world_size > 1:
            torch.distributed.all_reduce(output_, group=tp_group)

        if bias is not None:
            output_ = output_ + bias
        return output_, input_, weight
