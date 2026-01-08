import torch

from mindspeed.te.pytorch.fp8 import fp8_matmul
from mindspeed.te.pytorch.module.ops.comm_overlap_ops import CommOverlapOps


class DefaultOps(CommOverlapOps):

    @staticmethod
    def allgather_matmul(input_, weight, bias, fp8_meta, key=None, fp8_enable=False, transpose=(False, False)):

        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * fp8_meta.tp_world_size

        total_input = torch.empty(dim_size, dtype=input_.dtype, device=input_.device)
        torch.distributed._all_gather_base(total_input, input_.contiguous(), group=fp8_meta.tp_group, async_op=False)

        if not fp8_enable:
            output = torch.matmul(
                total_input.t() if transpose[0] else total_input,
                weight.t() if transpose[1] else weight
            )
            return output, total_input, None
        else:
            output, input_fp8, weight_fp8 = fp8_matmul(total_input, weight, fp8_meta, key, transpose)
            return output, input_fp8, weight_fp8

    @staticmethod
    def matmul_reduce_scatter(input_, weight, bias, fp8_meta, key=None, fp8_enable=False, transpose=(False, True)):
        if not fp8_enable:
            output_ = torch.matmul(
                input_.t() if transpose[0] else input_,
                weight.t() if transpose[1] else weight
            )
        else:
            output_, input_, weight = fp8_matmul(input_, weight, fp8_meta, key, transpose)

        dim_size = list(output_.size())
        dim_size[0] = dim_size[0] // fp8_meta.tp_world_size
        output = torch.empty(dim_size, dtype=output_.dtype, device=torch.cuda.current_device())

        torch.distributed._reduce_scatter_base(output, output_.contiguous(), group=fp8_meta.tp_group)
        return output, input_, weight

    @staticmethod
    def matmul_all_reduce(input_, weight, bias, fp8_meta, key=None, fp8_enable=False):

        if not fp8_enable:
            output_ = torch.matmul(input_, weight.t())
        else:
            output_, input_, weight = fp8_matmul(input_, weight, fp8_meta, key, (False, True))

        if fp8_meta.tp_world_size > 1:
            torch.distributed.all_reduce(output_, group=fp8_meta.tp_group)

        if bias is not None:
            output_ = output_ + bias
        return output_, input_, weight
