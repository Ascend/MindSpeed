# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.


"""MoE Permutaion API"""
from typing import Tuple, Optional

import torch
import torch_npu


class MoePermuteMaskMap(torch.autograd.Function):
    """functional Permute with mask router map"""
    @staticmethod
    def forward(
            ctx,
            tokens,
            routing_map,
            probs: Optional[torch.Tensor] = None,
            num_out_tokens: Optional[int] = None,
            drop_and_pad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        permuted_input, permuted_probs, sorted_indices = torch_npu.npu_moe_token_permute_with_routing_map(
            tokens, routing_map, probs=probs, num_out_tokens=num_out_tokens, drop_and_pad=drop_and_pad)

        num_tokens, _ = tokens.shape
        num_experts = routing_map.shape[1]
        ctx.num_tokens = num_tokens
        ctx.num_experts = num_experts
        ctx.drop_and_pad = drop_and_pad
        ctx.sorted_indices = sorted_indices
        ctx.routing_map = routing_map
        return permuted_input, permuted_probs, sorted_indices

    @staticmethod
    def backward(
            ctx,
            permuted_act_grad: torch.Tensor,
            permuted_probs_grad: torch.Tensor,
            _,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            sorted_indices = ctx.sorted_indices
            routing_map = ctx.routing_map
            num_tokens = ctx.num_tokens
            num_experts = ctx.num_experts
            drop_and_pad = ctx.drop_and_pad

            act_grad, probs_grad = torch_npu.npu_moe_token_permute_with_routing_map_grad(
                permuted_act_grad,
                permuted_probs_grad,
                sorted_indices,
                routing_map,
                num_experts,
                num_tokens,
                drop_and_pad
            )
        if not ctx.needs_input_grad[2]:
            probs_grad = None
        return act_grad, None, probs_grad, None, None


class MoeUnpermuteMaskMap(torch.autograd.Function):
    """functional Unpermute with mask router map"""

    @staticmethod
    def forward(
            ctx,
            permuted_tokens: torch.Tensor,
            sorted_indices: torch.Tensor,
            restore_shape: torch.Size,
            probs: Optional[torch.Tensor] = None,
            routing_map: Optional[torch.Tensor] = None,
            drop_and_pad: bool = False,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if not permuted_tokens.numel():
            ctx.probs = probs
            return permuted_tokens

        unpermuted_output, out_index, permuted_token_id, _ = torch_npu._npu_moe_token_unpermute_with_routing_map(
            permuted_tokens, sorted_indices, restore_shape, probs=probs, routing_map=routing_map,
            drop_and_pad=drop_and_pad)
        with_probs = probs is not None
        if with_probs:
            ctx.save_for_backward(permuted_tokens, probs)
        ctx.restore_shape = restore_shape
        ctx.drop_and_pad = drop_and_pad
        ctx.sorted_indices = sorted_indices
        ctx.routing_map = routing_map
        ctx.out_index = out_index
        ctx.permuted_token_id = permuted_token_id
        ctx.with_probs = with_probs

        return unpermuted_output

    @staticmethod
    def backward(ctx, unpermuted_tokens_grad):
        # pylint: disable=missing-function-docstring
        if not unpermuted_tokens_grad.numel():
            return unpermuted_tokens_grad, None, None, ctx.probs, None, None
        restore_shape = ctx.restore_shape
        drop_and_pad = ctx.drop_and_pad
        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            permuted_tokens, probs = None, None
            if ctx.with_probs:
                permuted_tokens, probs = ctx.saved_tensors
            sorted_indices = ctx.sorted_indices
            routing_map = ctx.routing_map
            out_index = ctx.out_index
            permuted_token_id = ctx.permuted_token_id
            if drop_and_pad:
                act_grad, probs_grad = (
                    torch_npu.npu_moe_token_unpermute_with_routing_map_grad(
                        unpermuted_tokens_grad,
                        out_index,
                        permuted_token_id=permuted_token_id,
                        routing_map=routing_map,
                        permuted_tokens=permuted_tokens,
                        probs=probs,
                        drop_and_pad=drop_and_pad,
                        restore_shape=restore_shape
                    )
                )

            else:
                act_grad, probs_grad = (
                    torch_npu.npu_moe_token_unpermute_with_routing_map_grad(
                        unpermuted_tokens_grad,
                        sorted_indices,
                        sorted_indices,
                        routing_map=routing_map,
                        permuted_tokens=permuted_tokens,
                        probs=probs,
                        drop_and_pad=drop_and_pad,
                        restore_shape=restore_shape
                    )
                )

        if not ctx.needs_input_grad[3]:
            probs_grad = None
        return act_grad, None, None, probs_grad, None, None
