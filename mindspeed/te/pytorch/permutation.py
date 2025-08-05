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
    ) -> torch.Tensor:

        permuted_input, permuted_probs, sorted_indices = torch_npu.npu_moe_token_permute_with_routing_map(
            tokens, routing_map, probs=probs, num_out_tokens=num_out_tokens, drop_and_pad=drop_and_pad)
        ctx.save_for_backward(sorted_indices, routing_map)

        num_tokens, _ = tokens.shape
        num_experts = routing_map.shape[1]
        ctx.num_tokens = num_tokens
        ctx.num_experts = num_experts
        ctx.drop_and_pad = drop_and_pad
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
            sorted_indices, routing_map = ctx.saved_tensors
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
            probs: torch.Tensor = None,
            routing_map: torch.Tensor = None,
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
            ctx.save_for_backward(permuted_tokens, sorted_indices, probs, routing_map, out_index, permuted_token_id)
        else:
            ctx.save_for_backward(sorted_indices, routing_map, out_index, permuted_token_id)
        ctx.restore_shape = restore_shape
        ctx.drop_and_pad = drop_and_pad
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
            if ctx.with_probs:
                permuted_tokens, sorted_indices, probs, routing_map, out_index, permuted_token_id = ctx.saved_tensors
            else:
                sorted_indices, routing_map, out_index, permuted_token_id = ctx.saved_tensors
            if drop_and_pad:
                if ctx.with_probs:
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
                            out_index,
                            permuted_token_id=permuted_token_id,
                            routing_map=routing_map,
                            permuted_tokens=None,
                            probs=None,
                            drop_and_pad=drop_and_pad,
                            restore_shape=restore_shape
                        )
                    )
            else:
                if ctx.with_probs:
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
                else:
                    act_grad, probs_grad = (
                        torch_npu.npu_moe_token_unpermute_with_routing_map_grad(
                            unpermuted_tokens_grad,
                            sorted_indices,
                            sorted_indices,
                            routing_map=routing_map,
                            permuted_tokens=None,
                            probs=None,
                            drop_and_pad=drop_and_pad,
                            restore_shape=restore_shape
                        )
                    )
        if not ctx.needs_input_grad[3]:
            probs_grad = None
        return act_grad, None, None, probs_grad, None, None
