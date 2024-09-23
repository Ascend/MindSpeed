# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import torch.nn as nn
import torch.distributed as dist


def get_range_list_of_3dshape(dim_size, world_size, kernel_size, stride):
    def find_last_le_k(arr, k):
        return max((element for element in arr if element < k), default=arr[-1])

    def find_first_ge_k(arr, k):
        return next((element for element in arr if element >= k), arr[-1])

    range_list = []
    stride_index = [i for i in range(0, dim_size, stride)]
    for rank in range(world_size):
        depth_per_sp = dim_size // world_size
        start_idx = find_first_ge_k(stride_index, rank * depth_per_sp)
        last_idx = find_last_le_k(stride_index, (rank + 1) * depth_per_sp) + 1
        end_idx = last_idx + kernel_size - 1 if rank < world_size - 1 else dim_size

        range_list.append([start_idx, end_idx])
    return range_list


def _split(input_, pg: dist.ProcessGroup, dim=-1, kernel_size=1, stride=1, depth_range=None):
    # skip if only one rank involved
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if world_size == 1:
        return input_

    if depth_range:
        start_idx, end_idx = depth_range[rank]
        output = input_[:, :, start_idx:end_idx, :, :].contiguous()
        return output, None

    # Split along last dimension.
    dim_size = input_.size(dim)

    start_end_idx_list = get_range_list_of_3dshape(dim_size, world_size, kernel_size, stride)
    start_idx, end_idx = start_end_idx_list[rank]
    output = input_[:, :, start_idx:end_idx, :, :].contiguous()

    return output, start_end_idx_list


def _gather(input_, pg: dist.ProcessGroup, total_depth, dim=2, kernel_size=1, stride=1, is_forward=True):
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)
    padding = 0 # not support padding currently

    # skip if only one rank involved
    if world_size == 1:
        return input_

    tensor_list = []
    start_end_idx_list = get_range_list_of_3dshape(total_depth, world_size, kernel_size, stride)
    original_start_end_idx_list = []
    conv_start_end_idx_list = []

    if is_forward:
        # forward: build the shapes after conv
        last_end_idx = 0
        for start_idx, end_idx in start_end_idx_list:
            length = end_idx - start_idx
            # O = (W-K+2P)/S + 1
            length = (length - kernel_size + 2 * padding) // stride + 1
            conv_start_end_idx_list.append([last_end_idx, last_end_idx + length])
            last_end_idx = last_end_idx + length
            tensor_list.append(torch.empty_like(input_[:, :, 0:1, :, :].expand(-1, -1, length, -1, -1)))
            output_start_end_idx_list = conv_start_end_idx_list
    else:
        # backward: build the original shapes before conv
        for start_idx, end_idx in start_end_idx_list:
            # O = (W-K+2P)/S + 1
            original_start_end_idx_list.append([start_idx, end_idx])
            tensor_list.append(torch.empty_like(input_[:, :, 0:1, :, :].expand(-1, -1, end_idx - start_idx, -1, -1)))
            output_start_end_idx_list = original_start_end_idx_list

    dist.all_gather(tensor_list, input_, group=pg)
    output = torch.cat(tensor_list, dim=dim).contiguous()
    if not is_forward:
        real_output = torch.zeros_like(input_[:, :, 0:1, :, :].expand(-1, -1, total_depth, -1, -1))
        for tensor, idx in zip(tensor_list, output_start_end_idx_list):
            start_idx, end_idx = idx
            for i in range(start_idx, end_idx):
                j = i - start_idx
                real_output[:, :, i, :, :] = real_output[:, :, i, :, :] + tensor[:, :, j, :, :]

        output = real_output
    return output, output_start_end_idx_list


class _ConvGatherForwardSplitBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, process_group, total_depth, dim, kernel_size, stride):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        output, depth_range = _gather(input_, process_group, total_depth, dim, kernel_size, stride, True)
        ctx.depth_range = depth_range
        return output


    @staticmethod
    def backward(ctx, grad_output):
        output, _ = _split(grad_output, ctx.mode, ctx.dim, ctx.kernel_size, ctx.stride, ctx.depth_range)
        return output, None, None, None, None, None, None


class _ConvSplitForwardGatherBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, process_group, dim, kernel_size, stride):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.total_depth = input_.shape[dim]
        output, _ = _split(input_, process_group, dim, kernel_size, stride)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, _ = _gather(grad_output, ctx.mode, ctx.total_depth, ctx.dim, ctx.kernel_size, ctx.stride, False)
        return output, None, None, None, None, None, None


class AllReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, conv3d_module, param_async, grad_reduce_handles):
        ctx.grad_reduce_handles = grad_reduce_handles
        ctx.param_async = param_async
        ctx.conv3d = conv3d_module
        return input

    @staticmethod
    def backward(ctx, grad_output):
        for param in ctx.conv3d.parameters():
            if param.grad is not None:
                if ctx.param_async:
                    handle = torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM, async_op=True)
                    ctx.grad_reduce_handles.append(handle)
                else:
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
        return grad_output, None, None, None


class Conv3DSequenceParallel(nn.Module):
    def __init__(self,
                 pg: dist.ProcessGroup,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 dilation=1,
                 bias=True,
                 param_async=False,
                 dtype=torch.bfloat16,
                 sp_size=1):
        super(Conv3DSequenceParallel, self).__init__()
        self.sp_size = sp_size
        self.depth_kernel_size = kernel_size[0]
        self.depth_stride = stride[0]
        self.param_async = param_async
        self.padding = 0  # not support padding currently
        self.pg = pg
        self.world_size = dist.get_world_size(pg)
        self.grad_reduce_handles = []

        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
            bias=bias
        ).npu().to(dtype)

    def forward(self, x):
        depth = x.shape[2]  # [batch_size, in_channels, depth, height, width]

        # O = (W-K+2P)/S + 1
        depth_after_conv = (depth - self.depth_kernel_size + 2 * self.padding) // self.depth_stride + 1
        if self.sp_size > 1 and (depth_after_conv // self.world_size) > 0:
            x = AllReduceFunction.apply(x, self.conv3d, self.param_async, self.grad_reduce_handles)
            x = _ConvSplitForwardGatherBackward.apply(x, self.pg, 2, self.depth_kernel_size, self.depth_stride)

        x = self.conv3d(x)

        if self.sp_size > 1 and (depth_after_conv // self.world_size) > 0:
            x = _ConvGatherForwardSplitBackward.apply(x, self.pg, depth, 2, self.depth_kernel_size, self.depth_stride)

        return x

    def get_param_grad_reduce_handles(self):
        return self.grad_reduce_handles

    def wait_param_grad_reduce_handles(self):
        for handle in self.grad_reduce_handles:
            handle.wait()
        self.grad_reduce_handles = []
