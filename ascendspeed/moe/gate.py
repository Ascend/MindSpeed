# Copyright (c) Microsoft Corporation.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py
# reworked/refactored some parts to make it run.
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .utils import gumbel_rsample, _capacity, einsum, _top_idx, _one_hot_to_float, MoEAuxLossAutoScaler
from .config import Config


class TopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    weight: torch.nn.Linear

    def __init__(self, config: Config) -> None:
        super().__init__()

        # Only top-1 and top-2 are supported at the moment.
        if config.topk != 1 and config.topk != 2:
            raise ValueError('Only top-1 and top-2 gatings are supported.')
        self.weight = torch.nn.Linear(config.hidden_size, config.num_experts, bias=False).float()
        self.config = config

    def forward(self, input: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:  # type: ignore
        if self.weight.weight.dtype != torch.float32:
            self.weight = self.weight.float()
        input_fp32 = input.float()
        logits = self.weight(input_fp32)

        if self.config.topk == 1:
            gate_output = top1gating(logits, self.config)
        else:
            gate_output = top2gating(logits, self.config)

        return gate_output


def top1gating(logits: Tensor, config: Config) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    if config.noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)
    capacity = _capacity(gates, torch.tensor(config.capacity_factor), torch.tensor(config.min_capacity))

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(logits_w_noise if config.noisy_gate_policy == 'RSample' else gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts

    # capacity
    top_idx = _top_idx(mask1, capacity)
    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1

    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float

    # Compute locations in capacity dim
    locations1 = torch.cumsum(mask1, dim=0) - 1
    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    combine_weights = einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


def apply_aux_loss(config, gates, mask1):
    num_experts = int(gates.shape[1])
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts
    if config.aux_loss_coef > 0:
        l_aux = l_aux * config.aux_loss_coef
        gates = MoEAuxLossAutoScaler.apply(gates, l_aux)
    return gates, l_aux

def apply_z_loss(config, logits):
    """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
    """
        if config.z_loss_coef > 0:
            z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * config.z_loss_coef
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
        return logits


def top2gating(logits: Tensor, config: Config) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # apply z loss
    logits = apply_z_loss(config, logits)

    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)
    num_experts = int(gates.shape[1])

    capacity = _capacity(gates, torch.tensor(config.capacity_factor * 2), torch.tensor(config.min_capacity))

    _, selected_experts = torch.topk(gates, config.topk, dim=-1)
    mask = F.one_hot(selected_experts, num_classes=num_experts)
    mask1 = mask[:, 0, :]
    mask2 = mask[:, 1, :]

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')

    # Compute l_aux
    gates, l_aux = apply_aux_loss(config, gates, mask1)

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts
