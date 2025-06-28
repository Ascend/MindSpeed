# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import torch_npu

from mindspeed.te.module.ops.comm_overlap_ops import CommOverlapOps, COMM_OVERLAP_CONFIG


class Mc2Ops(CommOverlapOps):

    @staticmethod
    def allgather_matmul(input_, weight, bias, fp8_meta=None, key=None, fp8_enable=False):
        world_size = COMM_OVERLAP_CONFIG.get_tp_size()
        hcomm_info = get_hcomm_info()

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])
        output, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
            x,
            weight,
            hcomm_info,
            world_size,
            bias=bias,
            gather_index=0,
        )
        output = output.view(int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1])
        return output, all_gather_grad_output, None

    @staticmethod
    def matmul_reduce_scatter(input_, weight, bias, fp8_meta=None, key=None, fp8_enable=False):
        world_size = COMM_OVERLAP_CONFIG.get_tp_size()
        hcomm_info = get_hcomm_info()

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])
        output = torch_npu.npu_mm_reduce_scatter_base(
            x, weight.t(), hcomm_info, world_size, reduce_op="sum", bias=bias
        )
        output = output.view(
            int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1]
        )
        return output, None, None

    @staticmethod
    def matmul_all_reduce(input_, weight, bias, fp8_meta=None, key=None, fp8_enable=False):
        world_size = COMM_OVERLAP_CONFIG.get_tp_size()
        hcomm_info = get_hcomm_info()

        x = input_.reshape(input_.shape[0] * input_.shape[1], input_.shape[2])
        output = torch_npu.npu_mm_all_reduce_base(
            x, weight.t(), hcomm_info, reduce_op="sum", bias=bias
        )
        output = output.view(
            int(output.shape[0] / input_.shape[1]), input_.shape[1], output.shape[1]
        )
        return output, None, None


def get_hcomm_info():
    group = COMM_OVERLAP_CONFIG.get_tp_group()
    rank = torch.distributed.get_rank(group)
    hcomm_info = None
    if torch.__version__ > "2.0":
        global_rank = torch.distributed.get_global_rank(group, rank)
        hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
            global_rank
        )
    else:
        hcomm_info = group.get_hccl_comm_name(rank)
    return hcomm_info