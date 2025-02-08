# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm
from mindspeed.ops.npu_bmm_reduce_scatter_all_to_all import npu_bmm_reducescatter_alltoall


def grouped_gemm_is_available():
    try:
        from mindspeed.ops.gmm import npu_gmm
        return True
    except ImportError:
        return False


def assert_grouped_gemm_is_available():
    if not grouped_gemm_is_available():
        raise ImportError("from mindspeed.ops.gmm import npu_gmm failed.")


class Ops:
    @staticmethod
    def gmm(a, b, batch_sizes, trans_b=False, gemm_fusion=False, original_weight=None):
        from mindspeed.ops.gmm import npu_gmm

        if trans_b:
            b = b.t()
        group_list = torch.cumsum(batch_sizes, dim=0)
        return npu_gmm(a, b, bias=None, group_list=group_list, group_type=0, gemm_fusion=gemm_fusion, original_weight=original_weight)


ops = Ops


def get_device_capability():
    return 9, 0


def get_hcomm_info_world(comm_group):
    rank = torch.distributed.get_rank()
    hcomm_info = None

    if torch.__version__ > "2.0.1":
        hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = comm_group.get_hccl_comm_name(rank)
    return hcomm_info


class FusedAllgatherBmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, bmm_param):

        group_ep = bmm_param['group_ep']
        group_tp = bmm_param['group_tp']
        need_recompute = bmm_param['need_recompute']
        shard_type = bmm_param['shard_type']

        ep_size = torch.distributed.get_world_size(group=group_ep)
        tp_size = torch.distributed.get_world_size(group=group_tp)

        tp_group_hcomm = get_hcomm_info_world(group_tp)
        ep_group_hcomm = get_hcomm_info_world(group_ep)

        out = npu_alltoall_allgather_bmm(
            input_, weight, ep_group_hcomm, ep_size, tp_group_hcomm, tp_size, bias=bias, shard_type=shard_type,
            act_type="None", need_allgather_out=True, need_activation_feature=False
        )
        bmm_out = out[0]
        allgather_out = out[1]

        if need_recompute:
            ctx.save_for_backward(input_, weight)
        else:
            ctx.save_for_backward(allgather_out, weight)

        ctx.bias = bias
        ctx.need_recompute = need_recompute
        ctx.group_ep = ep_group_hcomm
        ctx.group_tp = tp_group_hcomm
        ctx.ep_size = ep_size
        ctx.tp_size = tp_size
        ctx.shard_type = shard_type
        return bmm_out

    @staticmethod
    def backward(ctx, grad_output):

        need_recompute = ctx.need_recompute
        bias = ctx.bias
        group_ep = ctx.group_ep
        group_tp = ctx.group_tp
        ep_size = ctx.ep_size
        tp_size = ctx.tp_size
        shard_type = ctx.shard_type

        allgather_out = None
        input_ = None

        if need_recompute:
            input_, weight = ctx.saved_tensors
        else:
            allgather_out, weight = ctx.saved_tensors

        if need_recompute:
            out = npu_alltoall_allgather_bmm(
                input_, weight, group_ep, ep_size, group_tp, tp_size, bias=bias, shard_type=shard_type,
                act_type="None", need_allgather_out=True, need_activation_feature=False
            )
            allgather_out = out[1]

        # b,m,k @ b,k,n -> b,m,n
        # dx: b,m,n @ (b,k,n).t() -> b,m,k
        out = npu_bmm_reducescatter_alltoall(
            grad_output, weight.transpose(-1, -2), group_ep, ep_size, group_tp, tp_size,
            bias=None, shard_type=shard_type
        )

        # b,m,k @ b,k,n -> b,m,n
        # dw: (b,m,k).t() @ (b,m,n).t() -> b,k,n
        grad_bmm_w = torch.bmm(allgather_out.transpose(-1, -2), grad_output)
        grad_bias = None
        if bias is not None:
            grad_bias = torch.sum(grad_output, dim=-1)

        return out, grad_bmm_w, grad_bias, None


class FusedBmmReduceScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, weight, bias, bmm_param):

        group_ep = bmm_param['group_ep']
        group_tp = bmm_param['group_tp']
        shard_type = bmm_param['shard_type']

        ep_size = torch.distributed.get_world_size(group=group_ep)
        tp_size = torch.distributed.get_world_size(group=group_tp)

        tp_group_hcomm = get_hcomm_info_world(group_tp)
        ep_group_hcomm = get_hcomm_info_world(group_ep)

        out = npu_bmm_reducescatter_alltoall(
            input_, weight, ep_group_hcomm, ep_size, tp_group_hcomm, tp_size,
            bias=bias, shard_type=shard_type
        )

        ctx.save_for_backward(input_, weight)

        ctx.bias = bias
        ctx.group_ep = ep_group_hcomm
        ctx.group_tp = tp_group_hcomm
        ctx.ep_size = ep_size
        ctx.tp_size = tp_size
        ctx.shard_type = shard_type
        return out

    @staticmethod
    def backward(ctx, grad_output):

        bias = ctx.bias
        group_ep = ctx.group_ep
        group_tp = ctx.group_tp
        ep_size = ctx.ep_size
        tp_size = ctx.tp_size
        shard_type = ctx.shard_type

        input_, weight = ctx.saved_tensors

        # b,m,k @ b,k,n -> b,m,n
        # dx: b,m,n @ (b,k,n).t() -> b,m,k
        out = npu_alltoall_allgather_bmm(
            grad_output, weight.transpose(-1, -2), group_ep, ep_size, group_tp, tp_size,
            bias=bias, shard_type=shard_type, act_type="None", need_allgather_out=True, need_activation_feature=False
        )
        bmm_out_grad = out[0]
        allgather_out_grad = out[1]

        # b,m,k @ b,k,n -> b,m,n
        # dw: (b,m,k).t() @ (b,m,n).t() -> b,k,n
        grad_bmm_w = torch.bmm(input_.transpose(-1, -2), allgather_out_grad)
        grad_bias = None
        if bias is not None:
            grad_bias = torch.sum(grad_output, dim=-1)

        return bmm_out_grad, grad_bmm_w, grad_bias, None


def fused_alltoall_gather_bmm(input_, weight, bias, bmm_param):
    return FusedAllgatherBmmFunction.apply(input_, weight, bias, bmm_param)


def fused_bmm_reducescatter_alltoall(input_, weight, bias, bmm_param):
    return FusedBmmReduceScatterFunction.apply(input_, weight, bias, bmm_param)
