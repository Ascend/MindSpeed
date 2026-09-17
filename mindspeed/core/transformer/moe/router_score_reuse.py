# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Reuse router scores without changing Megatron's numerical behavior."""

from functools import wraps
from typing import Optional, Tuple

import torch

from mindspeed.args_utils import get_full_args
from mindspeed.core.transformer.moe.moe_utils import get_fixed_router_indices
from megatron.core.transformer.moe.moe_utils import (
    apply_router_token_dropping,
    group_limited_topk,
)


class _DualRouterScore(torch.autograd.Function):
    """Evaluate a score function once while preserving its two backward branches."""

    SOFTMAX = 0
    SIGMOID = 1

    @staticmethod
    def forward(ctx, logits: torch.Tensor, score_function: int):
        if score_function == _DualRouterScore.SOFTMAX:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        elif score_function == _DualRouterScore.SIGMOID:
            # Megatron explicitly promotes sigmoid router logits before both
            # the dispatch and auxiliary branches.
            scores = torch.sigmoid(logits.float())
        else:
            raise ValueError(f"Unsupported reusable score function: {score_function}")

        ctx.score_function = score_function
        ctx.input_dtype = logits.dtype
        ctx.save_for_backward(scores)
        ctx.set_materialize_grads(False)
        # A custom Function identifies differentiable outputs by output_nr.
        # Returning the exact same Tensor object twice assigns both users to
        # one output slot and merges/misroutes their gradients. A view creates
        # distinct output slots while still sharing the score storage.
        return scores, scores.view_as(scores)

    @staticmethod
    def backward(ctx, dispatch_grad: Optional[torch.Tensor], aux_grad: Optional[torch.Tensor]):
        (scores,) = ctx.saved_tensors

        def score_backward(grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if grad is None:
                return None
            if ctx.score_function == _DualRouterScore.SOFTMAX:
                # ``softmax(logits, dtype=torch.float32)`` first differentiates
                # in FP32, then casts each independent branch back to the
                # logits dtype before autograd accumulates the two branches.
                # Keep that cast before the addition below so BF16 rounding
                # occurs at the same point as in Megatron's original graph.
                return torch.ops.aten._softmax_backward_data.default(grad, scores, -1, scores.dtype).to(ctx.input_dtype)
            # In Megatron's original graph, each sigmoid branch traverses its
            # own ``logits.float()`` node. Convert each branch independently
            # before accumulating them so BF16 rounding happens in the same
            # order as the unfused graph.
            return torch.ops.aten.sigmoid_backward.default(grad, scores).to(ctx.input_dtype)

        dispatch_logits_grad = score_backward(dispatch_grad)
        aux_logits_grad = score_backward(aux_grad)
        if dispatch_logits_grad is None:
            logits_grad = aux_logits_grad
        elif aux_logits_grad is None:
            logits_grad = dispatch_logits_grad
        else:
            # Megatron originally has two independent score nodes whose gradients
            # are accumulated at logits. Keep the two derivatives separate before
            # adding them, instead of differentiating one score with a summed grad.
            logits_grad = dispatch_logits_grad + aux_logits_grad
        return logits_grad, None


def _compute_topk(
    scores: torch.Tensor,
    topk: int,
    num_groups: Optional[int],
    group_topk: Optional[int],
    router_replay,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = scores.shape

    def native_topk(
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if group_topk:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        return torch.topk(
            scores,
            k=topk,
            dim=1,
            sorted=torch.is_grad_enabled(),
        )

    if router_replay is None:
        return native_topk(scores, topk, num_groups, group_topk)
    return router_replay.get_replay_topk(scores, topk, num_groups, group_topk, native_topk)


def _build_routing_tensors(
    logits: torch.Tensor, probs: torch.Tensor, top_indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = logits.shape[0]
    if torch.are_deterministic_algorithms_enabled():
        rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
        routing_probs = torch.zeros_like(logits)
        routing_probs.index_put_((rows, top_indices), probs, accumulate=False)
        routing_map = torch.zeros_like(logits, dtype=logits.dtype)
        routing_map.index_put_(
            (rows, top_indices),
            torch.ones_like(probs, dtype=routing_map.dtype),
            accumulate=False,
        )
        routing_map = routing_map.bool()
    else:
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_probs, routing_map


def topk_routing_with_aux_score_reuse(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
    router_replay=None,
    padding_mask: Optional[torch.Tensor] = None,
    fixed_router: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dispatch and auxiliary routing from one score-function evaluation.

    This helper intentionally supports only the cases selected by
    :func:`router_score_reuse_wrapper`. Unsupported configurations stay on
    Megatron's original implementation.
    """
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    if logits.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Router score reuse requires FP32 or BF16 logits for strict numerical alignment.")

    if score_function == "softmax":
        if not use_pre_softmax:
            raise ValueError("Softmax router score reuse requires pre-softmax routing.")
        dispatch_scores, scores_for_aux_loss = _DualRouterScore.apply(logits, _DualRouterScore.SOFTMAX)
        probs, dynamic_top_indices = _compute_topk(dispatch_scores, topk, num_groups, group_topk, router_replay)
    elif score_function == "sigmoid":
        dispatch_scores, aux_scores = _DualRouterScore.apply(logits, _DualRouterScore.SIGMOID)
        if expert_bias is not None:
            scores_for_routing = dispatch_scores + expert_bias.float()
            _, dynamic_top_indices = _compute_topk(scores_for_routing, topk, num_groups, group_topk, router_replay)
            selected_scores = torch.gather(dispatch_scores, dim=1, index=dynamic_top_indices)
        else:
            selected_scores, dynamic_top_indices = _compute_topk(
                dispatch_scores, topk, num_groups, group_topk, router_replay
            )
        probs = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else selected_scores
        scores_for_aux_loss = aux_scores / (aux_scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Unsupported reusable score function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor
    probs = probs.type_as(logits)

    assignment_indices = get_fixed_router_indices(logits, topk) if fixed_router else dynamic_top_indices
    routing_probs, routing_map = _build_routing_tensors(logits, probs, assignment_indices)

    # With plain pre-softmax routing, dispatch and auxiliary routing select the
    # same indices. Group routing and replay deliberately use different Top-K
    # rules for dispatch, so their auxiliary Top-K must still be executed.
    can_reuse_routing_map = (
        score_function == "softmax" and group_topk is None and expert_bias is None and router_replay is None
    )
    if can_reuse_routing_map:
        routing_map_for_aux_loss = routing_map
    else:
        _, aux_top_indices = torch.topk(scores_for_aux_loss, k=topk, dim=1)
        aux_assignment_indices = assignment_indices if fixed_router else aux_top_indices
        routing_map_for_aux_loss = torch.zeros_like(logits).int().scatter(1, aux_assignment_indices, 1).bool()

    if padding_mask is not None:
        valid_mask = (~padding_mask).unsqueeze(-1)
        routing_map_for_aux_loss = routing_map_for_aux_loss * valid_mask
        scores_for_aux_loss = scores_for_aux_loss * valid_mask

    return routing_probs, routing_map, routing_map_for_aux_loss, scores_for_aux_loss


def _get_runtime_args():
    try:
        return get_full_args()
    except (AttributeError, RuntimeError):
        return None


def router_score_reuse_wrapper(routing):
    """Fuse duplicate dispatch/auxiliary score work in ``TopKRouter.routing``."""

    @wraps(routing)
    def wrapper(self, logits: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        args = _get_runtime_args()
        if args is not None and getattr(args, "moe_tp_extend_ep", False):
            # The TP-extend replacement predates padding-mask support and has a
            # different signature. Preserve that replacement exactly.
            return routing(self, logits)

        can_reuse_score = (
            args is not None
            and self.training
            and torch.is_grad_enabled()
            and self.is_aux_loss_enabled()
            and self.routing_type != "sinkhorn"
            and not self.config.moe_router_fusion
            and logits.dtype in (torch.float32, torch.bfloat16)
            and self.score_function in ("softmax", "sigmoid")
            and (self.score_function != "softmax" or self.config.moe_router_pre_softmax)
        )
        if not can_reuse_score:
            return routing(self, logits, padding_mask=padding_mask)

        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1)

        logits = self.apply_z_loss(logits, padding_mask=padding_mask)
        probs, routing_map, routing_map_for_aux_loss, scores_for_aux_loss = topk_routing_with_aux_score_reuse(
            logits,
            self.topk,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            router_replay=self.router_replay,
            padding_mask=padding_mask,
            fixed_router=getattr(args, "fix_router", False),
        )

        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        probs = self._apply_aux_loss(
            probs,
            scores_for_aux_loss,
            routing_map_for_aux_loss,
            with_padding_mask=padding_mask is not None,
        )
        probs = self._apply_seq_aux_loss(
            probs,
            scores_for_aux_loss,
            routing_map_for_aux_loss,
            seq_length,
            bsz,
            with_padding_mask=padding_mask is not None,
        )
        probs = self._apply_global_aux_loss(
            probs,
            scores_for_aux_loss,
            routing_map_for_aux_loss,
            with_padding_mask=padding_mask is not None,
        )

        self._apply_expert_bias(routing_map, padding_mask=padding_mask)
        return probs, routing_map

    return wrapper
