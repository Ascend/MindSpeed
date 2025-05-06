# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
import torch_npu
from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import (
    set_swap_status, get_swap_status, 
    set_prob_backward_need_tensors, 
    get_swap_stream
)


class UnpermuteWithoutActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        restore_shape: torch.Size,
        probs: torch.Tensor = None,
        routing_map: torch.Tensor = None,

    ):
        """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens with their corresponding probabilities.
            Only used when zero_memory is setting.

        Args:
            permuted_tokens (torch.Tensor): The tensor of permuted tokens to be unpermuted.
            sorted_indices (torch.Tensor): The tensor of sorted indices used to unpermute the tokens.
            restore_shape (torch.Size): The input shape before permutation, only used in padding mode.
            probs (torch.Tensor): The tensor of probabilities corresponding to the permuted tokens. 
                If provided, the unpermuted tokens will be merged with their respective probabilities.
            routing_map (torch.Tensor): The mapping of tokens to experts, with shape
                [num_tokens, num_experts].
        
        Returns:
            torch.Tensor: The unpermuted tokens, optionally merged with probabilities.
        """

        if sorted_indices.numel() != permuted_tokens.size(0):
            raise AssertionError("")
        saved_tensors = [sorted_indices]

        with torch.no_grad():
            _, hidden = restore_shape
            if probs is not None:
                # Unpermute and merge the tokens with their probabilities
                assert routing_map is not None, "Mask must be provided to permute the probs."
                permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
                tensor_to_swap = permuted_tokens
                permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)
                saved_tensors.append(permuted_probs)
                ctx.topk = probs.size(1)
                ctx.probs_shape = probs.shape
                ctx.probs_dtype = probs.dtype
                ctx.hidden_size = hidden
                ctx.routing_map = routing_map
            else:
                # Unpermute the tokens without merge
                ctx.topk = 1
            ctx.save_for_backward(*saved_tensors)

            unpermuted_tokens = torch.zeros(
                restore_shape, device=permuted_tokens.device, dtype=permuted_tokens.dtype
            )

            ctx.permuted_tokens_shape = permuted_tokens.shape
            ctx.unpermuted_tokens_shape = unpermuted_tokens.shape

            unpermuted_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)

            if probs is not None:
                swap_stream, last_tensor = get_swap_status()
                if last_tensor is not None:
                    torch.npu.current_stream().wait_stream(swap_stream)
                    last_tensor.untyped_storage().resize_(0)
                forward_event = torch.npu.Event()
                forward_event.record()
                set_swap_status(tensor_to_swap)
                ctx.tensor_cpu = torch.empty(tensor_to_swap.shape, dtype=tensor_to_swap.dtype, pin_memory=True, device='cpu')
                with torch_npu.npu.stream(swap_stream):
                    swap_stream.wait_event(forward_event)
                    ctx.tensor_cpu.untyped_storage().copy_(tensor_to_swap.untyped_storage(), non_blocking=True)
                    ctx.swap_event = torch.npu.Event()
                    ctx.swap_event.record()

            ctx.matmul_output_shape = unpermuted_tokens.shape

        return unpermuted_tokens

    @staticmethod
    def backward(ctx, *args):
        if ctx.topk > 1:
            (indices, permuted_probs) = ctx.saved_tensors
        else:
            (indices,) = ctx.saved_tensors
        ctx.save_for_backward()

        if ctx.topk > 1:
            matmul_output_grad = args[0].expand(ctx.matmul_output_shape)
            backward_event1 = torch.npu.Event()
            backward_event1.record()
            swap_stream = get_swap_stream()
            permuted_tokens = torch.empty(ctx.tensor_cpu.shape, dtype=ctx.tensor_cpu.dtype, device=torch.npu.current_device())
            gather_indices = indices.unsqueeze(1).expand(-1, ctx.hidden_size)
            permuted_tokens_grad = torch.gather(matmul_output_grad, 0, gather_indices)
            #wait tensor_cpu swap_event.
            with torch_npu.npu.stream(swap_stream):
                swap_stream.wait_event(backward_event1)
                swap_stream.wait_event(ctx.swap_event)
                permuted_tokens.untyped_storage().copy_(ctx.tensor_cpu.untyped_storage(), non_blocking=True)
                ctx.tensor_cpu = None
            #Prepare for route_graph backward.
            set_prob_backward_need_tensors(permuted_tokens_grad, permuted_tokens)
            permuted_tokens_grad = permuted_tokens_grad * permuted_probs.unsqueeze(-1)

        else:
            permuted_tokens_grad = args[0]

        return permuted_tokens_grad, None, None, None, None
