# Copyright (c) 2022; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from megatron.training import get_args
from megatron.core.parallel_state import get_tensor_and_expert_parallel_group
from megatron.core.tensor_parallel.mappings import _reduce_scatter_along_first_dim_moe
from megatron.core.transformer.moe.moe_utils import topk_softmax_with_capacity


def _gather_along_first_dim_moe_async(input_, async_op):
    """Gather tensors and concatenate along the first dimension."""
    group = get_tensor_and_expert_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    handle = torch.distributed._all_gather_base(output, input_.contiguous(), group=group, async_op=async_op)

    return output, handle


class _GatherFromSequenceParallelRegionToMOEAsync(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_first_dim_moe_async(input_, async_op=True)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_first_dim_moe_async(input_, async_op=True)

    @staticmethod
    def backward(ctx, grad_output, grad_handle):
        return _reduce_scatter_along_first_dim_moe(grad_output)


def gather_from_sequence_parallel_region_to_moe_async(input_):
    return _GatherFromSequenceParallelRegionToMOEAsync.apply(input_)


def aux_loss_load_balancing(self, logits: torch.Tensor):
    probs, indices, tokens_per_expert = topk_softmax_with_capacity(
        logits,
        self.topk,
        capacity_factor=self.config.moe_expert_capacity_factor,
        pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        drop_policy=self.config.moe_token_drop_policy,
        use_pre_softmax=self.config.moe_router_pre_softmax,
    )
    global_indices = indices
    if self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1 and not get_args().tp_2d):
        with torch.no_grad():
            global_indices = gather_from_sequence_parallel_region_to_moe_async(indices)

    # Apply load balancing loss
    if self.training:
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        probs = self.apply_load_balancing_loss(scores, tokens_per_expert, activation=probs)
    return probs, global_indices


def routing_tp_extend_ep(self, logits: torch.Tensor):
    """Top-k routing function

    Args:
        logits (torch.Tensor): Logits tensor after gating.

    Returns:
        probs (torch.Tensor): the probabilities tensor after load balancing.
        indices (torch.Tensor): the indices tensor after top-k selection.
    """
    logits = logits.view(-1, self.config.num_moe_experts)

    # Apply Z-Loss
    logits = self.apply_z_loss(logits)

    if self.routing_type == "sinkhorn":
        scores, indices = self.sinkhorn_load_balancing(logits)
    elif self.routing_type == "aux_loss":
        scores, indices = self.aux_loss_load_balancing(logits)
    elif self.routing_type == "none":
        # A naive top-k routing without load balancing
        scores, indices, _ = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
        )
    else:
        raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

    return scores, indices
