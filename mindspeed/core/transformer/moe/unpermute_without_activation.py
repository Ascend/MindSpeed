# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.distributed
import torch.distributed as dist
import torch_npu
from megatron.training import get_args
from mindspeed.core.transformer.moe.moe_utils import (set_swap_status, get_swap_status,
                                                      set_prob_backward_need_tensors, get_swap_stream)


class UnpermuteWithoutActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                permuted_tokens: torch.Tensor,
                sorted_indices: torch.Tensor,
                probs: torch.Tensor = None,
                padded_mode: bool = False,
                restore_shape: torch.Size = None,
                ):
        """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens with their corresponding probabilities.

        Args:
            permuted_tokens (torch.Tensor): The tensor of permuted tokens to be unpermuted.
            sorted_indices (torch.Tensor): The tensor of sorted indices used to unpermute the tokens.
            probs (torch.Tensor, optional): The tensor of probabilities corresponding to the permuted tokens. If provided, the unpermuted tokens will be merged with their respective probabilities.
            padded_mode (bool, optional): If True, indicating the indices are padded to [num_expert, capacity] to denote selected tokens per expert. Defaults to False.
            restore_shape (torch.Size, optional): The input shape before permutation, only used in padding mode. Defaults to None.

        Returns:
            torch.Tensor: The unpermuted tokens, optionally merged with probabilities.
        """
        moe_hierarchical_alltoallv = get_args().moe_hierarchical_alltoallv
        if padded_mode:
            raise ValueError("moe-zero-memory temporally does not support padded mode")

        if sorted_indices.numel() != permuted_tokens.size(0):
            raise AssertionError("")
        saved_tensors = [sorted_indices]

        with torch.no_grad():
            if probs is not None:
                # Unpermute and merge the tokens with their probabilities
                num_unpermuted_tokens = probs.numel()
                saved_tensors.append(probs)
                ctx.topk = probs.size(1)
                ctx.probs_shape = probs.shape
                ctx.probs_dtype = probs.dtype
            else:
                # Unpermute the tokens without merge
                num_unpermuted_tokens = permuted_tokens.size(0)
                ctx.topk = 1
            ctx.save_for_backward(*saved_tensors)
            if moe_hierarchical_alltoallv:
                unpermuted_tokens = torch.zeros(
                    [ctx.topk * probs.shape[0], permuted_tokens.shape[-1]],
                    dtype=permuted_tokens.dtype,
                    device=permuted_tokens.device,
                )
                unpermuted_tokens = \
                    unpermuted_tokens.scatter(0, sorted_indices.unsqueeze(1).expand(-1, permuted_tokens.shape[1]),
                                              permuted_tokens)
            elif not get_args().use_fused_moe_token_permute_and_unpermute:
                unpermuted_tokens = torch.zeros(
                    [num_unpermuted_tokens, permuted_tokens.shape[-1]],
                    dtype=permuted_tokens.dtype,
                    device=permuted_tokens.device,
                )
                unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
            else:
                unpermuted_tokens = permuted_tokens.index_select(0, sorted_indices)

            ctx.permuted_tokens_shape = permuted_tokens.shape
            ctx.unpermuted_tokens_shape = unpermuted_tokens.shape
            unpermuted_tokens = unpermuted_tokens.reshape(-1, ctx.topk, permuted_tokens.size(-1))
            permuted_tokens.untyped_storage().resize_(0)

            if probs is not None:
                tensor_to_swap = unpermuted_tokens
                unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
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
            unpermuted_tokens = unpermuted_tokens.sum(dim=1)

        return unpermuted_tokens

    @staticmethod
    def backward(ctx, *args):
        moe_hierarchical_alltoallv = get_args().moe_hierarchical_alltoallv
        if ctx.topk > 1:
            (indices, probs) = ctx.saved_tensors
        else:
            (indices,) = ctx.saved_tensors
        ctx.save_for_backward()

        if ctx.topk > 1:
            matmul_output_grad = args[0].unsqueeze(dim=1).expand(ctx.matmul_output_shape)
            backward_event1 = torch.npu.Event()
            backward_event1.record()
            swap_stream = get_swap_stream()
            unpermuted_tokens = torch.empty(ctx.tensor_cpu.shape, dtype=ctx.tensor_cpu.dtype, device=torch.npu.current_device())
            with torch_npu.npu.stream(swap_stream):
                swap_stream.wait_event(backward_event1)
                swap_stream.wait_event(ctx.swap_event)
                unpermuted_tokens.untyped_storage().copy_(ctx.tensor_cpu.untyped_storage(), non_blocking=True)
                ctx.tensor_cpu = None
            unpermuted_tokens_grad = matmul_output_grad * probs.unsqueeze(-1)
            set_prob_backward_need_tensors(matmul_output_grad, unpermuted_tokens)
        else:
            unpermuted_tokens_grad = args[0]

        unpermuted_tokens_grad = unpermuted_tokens_grad.reshape(ctx.unpermuted_tokens_shape)

        if not get_args().use_fused_moe_token_permute_and_unpermute or moe_hierarchical_alltoallv:
            permuted_tokens_grad = unpermuted_tokens_grad.index_select(0, indices)
        else:
            permuted_tokens_grad = torch.zeros(size=ctx.permuted_tokens_shape, dtype=unpermuted_tokens_grad.dtype,
                                               device=torch.npu.current_device())
            permuted_tokens_grad.index_add_(0, indices, unpermuted_tokens_grad)

        return permuted_tokens_grad, None, None, None, None
