# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch


class CustomSliceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, split_point):
        ctx.save_for_backward(input_tensor)
        ctx.split_point = split_point

        permuted_local_tokens = input_tensor[:split_point].contiguous()
        permuted_remote_hot_tokens = input_tensor[split_point:].contiguous()

        return permuted_local_tokens, permuted_remote_hot_tokens

    @staticmethod
    def backward(ctx, grad_output_local, grad_output_remote):
        input_tensor, = ctx.saved_tensors
        grad_input = torch.zeros_like(input_tensor)

        split_point = ctx.split_point

        grad_input[:split_point] = grad_output_local
        grad_input[split_point:] = grad_output_remote

        return grad_input, None
