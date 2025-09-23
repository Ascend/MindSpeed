# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from mindspeed.core.transformer.moe.moe_feature import (
    topk_softmax_with_capacity,
    gather_from_sequence_parallel_region,
    _split_along_first_dim
)


def routing_tp_extend_ep(self, logits: torch.Tensor):
    """
    if use tp_extend_ep, logits is not need to gather from the tp region
    """
    logits = logits.view(-1, self.config.num_moe_experts)

    # Apply Z-Loss
    logits = self.apply_z_loss(logits)
    logits = gather_from_sequence_parallel_region(logits)

    if self.routing_type == "sinkhorn":
        scores, routing_map = self.sinkhorn_load_balancing(logits)
    elif self.routing_type == "aux_loss":
        scores, routing_map = self.aux_loss_load_balancing(logits)
    elif self.routing_type == "none":
        # A naive top-k routing without load balancing
        scores, routing_map, _ = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            deterministic_mode=self.config.deterministic_mode,
        )
    else:
        raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

    scores = _split_along_first_dim(scores)
    routing_map = _split_along_first_dim(routing_map)
    return scores, routing_map
