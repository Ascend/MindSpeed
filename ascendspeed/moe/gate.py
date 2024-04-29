# Copyright (c) Microsoft Corporation.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py
# reworked/refactored some parts to make it run.
from typing import Callable, Dict, Tuple, Union, Any
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from megatron.training import get_args

from .config import Config
from .utils import gumbel_rsample, _capacity, einsum, _one_hot_to_float, MoEAuxLossAutoScaler

exp_selection_uniform_map: Dict[torch.device, Callable] = {}


GatingTokenRearrangeInfo = namedtuple('GatingTokenRearrangeInfo', ['token_rearranged_ec_idx', 'token_exp_weights', 'expert_select_token_idx'])


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

    def forward(self, gate_input: torch.Tensor) -> Tuple[Tensor, ...]:  # type: ignore
        if self.weight.weight.dtype != torch.float32:
            self.weight = self.weight.float()
        input_fp32 = gate_input.float()
        logits = self.weight(input_fp32)

        if self.config.topk == 1:
            gate_output = top1gating(logits, self.config)
        else:
            gate_output = top2gating(logits, self.config)

        return gate_output


def top1gating(logits: Tensor, config: Config) -> Tuple[Tensor, ...]:
    """Implements Top1Gating on logits."""
    if config.noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    # token_sel_expert_weights: [S, E], ÿ��tokenѡ��ÿ��ר�ҵĸ���
    token_sel_expert_weights = F.softmax(logits, dim=1)
    capacity = _capacity(token_sel_expert_weights,
                        torch.tensor(config.capacity_factor),
                        torch.tensor(config.min_capacity))

    # Create a mask for 1st's expert per token
    # noisy gating
    final_logits = logits_w_noise if config.noisy_gate_policy == "RSample" else \
        token_sel_expert_weights
    # [S] ÿ��token��Ӧ��ר�ң�ȡ�������ģ�
    token_sel_expert_idx = torch.argmax(final_logits, dim=1)
    num_experts = int(token_sel_expert_weights.shape[1])
    token_sel_expert_mask = F.one_hot(token_sel_expert_idx, num_classes=num_experts)

    # Compute l_aux���ؾ���aux_loss
    me = torch.mean(token_sel_expert_weights, dim=0)
    ce = torch.mean(token_sel_expert_mask.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts
    all_args = get_args()
    # Random Token Selection(��tokenѡ��ר�ҵ�����0/1�����е�1ת��0~1֮���Ȩ��ֵ)
    if all_args.use_rts:  # default True.
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(
                low=torch.tensor(0.0, device=logits.device),
                high=torch.tensor(1.0, device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform
        # [S, E]
        token_sel_expert_score = token_sel_expert_mask * uniform(token_sel_expert_mask.shape)
    else:
        token_sel_expert_score = token_sel_expert_mask

    # ͨ��topCÿ��ר��ѡ������C��token��Ȼ���ԭʼ��mask1��ÿ��ר�ҿ���ѡ�񳬹�C��token��������ˣ�
    # ��������ר��������Ȩ�ص͵�token�����µõ� token_sel_expert_mask
    expert_sel_top_c_token_idx = torch.topk(token_sel_expert_score, k=capacity, dim=0)[1]
    token_sel_expert_mask *= torch.zeros_like(token_sel_expert_mask).scatter_(0, expert_sel_top_c_token_idx, 1)

    # Normalize gate probabilities
    token_sel_expert_mask_float = token_sel_expert_mask.float()
    token_sel_expert_weights = token_sel_expert_weights * token_sel_expert_mask_float

    token_idx_in_expert_with_noise = torch.cumsum(token_sel_expert_mask, dim=0) - 1
    masked_token_idx_in_expert = token_idx_in_expert_with_noise * token_sel_expert_mask
    token_offset_for_expert = torch.sum(masked_token_idx_in_expert, dim=1)
    if all_args.enable_token_rearrange_opt:
        # ���Ź��̣������ÿ��ר��ѡ���token��������expert_select_token_idx��shapeΪ: [E*C]
        # MoEǰ������и��ݴ�����ͨ��index_select APIʵ��token������
        # shape�仯���̣�[S, E]->[C, E]->[E, C]->[E*C]
        expert_sel_top_c_token_idx = torch.topk(token_sel_expert_mask,
                                                k=capacity,
                                                dim=0,
                                                sorted=True)[1]
        expert_select_token_idx = expert_sel_top_c_token_idx.t().reshape(config.num_experts * capacity)
        token_exp_weights, token_exp_idx = torch.max(token_sel_expert_weights, dim=1)
        token_rearranged_ec_idx = (capacity.to(torch.int32) * token_exp_idx.to(torch.int32) +
                                   token_offset_for_expert.to(torch.int32))
        top1_gating_token_infos = GatingTokenRearrangeInfo(token_rearranged_ec_idx=token_rearranged_ec_idx,
                                                           token_exp_weights=token_exp_weights,
                                                           expert_select_token_idx=expert_select_token_idx)
        return l_aux, top1_gating_token_infos
    else:
        token_locations_sc = _one_hot_to_float(token_offset_for_expert, capacity)
        combine_weights = einsum("se,sc->sec", token_sel_expert_weights, token_locations_sc)
        dispatch_mask = combine_weights.bool()
        return l_aux, combine_weights, dispatch_mask


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
    token_sel_expert_weights = F.softmax(logits, dim=1)
    num_experts = int(token_sel_expert_weights.shape[1])

    capacity = _capacity(token_sel_expert_weights,
                        torch.tensor(config.capacity_factor * 2),
                        torch.tensor(config.min_capacity))

    _, selected_experts = torch.topk(token_sel_expert_weights, config.topk, dim=-1)
    mask = F.one_hot(selected_experts, num_classes=num_experts)
    first_expert_mask = mask[:, 0, :]
    second_expert_mask = mask[:, 1, :]

    # Compute locations in capacity buffer
    locations_in_first_expert = torch.cumsum(first_expert_mask, dim=0) - 1
    locations_in_second_expert = torch.cumsum(second_expert_mask, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations_in_second_expert += torch.sum(first_expert_mask, dim=0, keepdim=True)

    # gating decisions
    token_sel_expert_weights, l_aux = apply_aux_loss(config, token_sel_expert_weights, first_expert_mask)

    # Remove locations outside capacity from mask
    first_expert_mask *= torch.lt(locations_in_first_expert, capacity)
    second_expert_mask *= torch.lt(locations_in_second_expert, capacity)

    # Store the capacity location for each token
    token_idx_in_first_expert = torch.sum(locations_in_first_expert * first_expert_mask, dim=1)
    token_idx_in_second_expert = torch.sum(locations_in_second_expert * second_expert_mask, dim=1)

    # Normalize gate probabilities
    first_expert_mask_float = first_expert_mask.float()
    second_expert_mask_float = second_expert_mask.float()
    token_first_exp_weights, token_first_exp_idx = torch.max(token_sel_expert_weights * first_expert_mask_float, dim=1)
    token_second_exp_weights, token_second_exp_idx = torch.max(token_sel_expert_weights * second_expert_mask_float,
                                                               dim=1)
    denom_s = token_first_exp_weights + token_second_exp_weights
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    token_first_exp_weights /= denom_s
    token_second_exp_weights /= denom_s
    all_args = get_args()
    if all_args.enable_token_rearrange_opt:
        token_rearranged_first_ec_idx = token_first_exp_idx.int() * capacity + token_idx_in_first_expert.int()
        token_rearranged_second_ec_idx = token_second_exp_idx.int() * capacity + token_idx_in_second_expert.int()
        # ���Ź��̣������ÿ��ר��ѡ���token��������expert_select_token_idx��shapeΪ: [E*C]
        # MoEǰ������и��ݴ�����ͨ��index_select APIʵ��token������
        # shape�仯���̣�[S, E]->[C, E]->[E, C]->[E*C]
        token_sel_first_exp_int_mask = first_expert_mask * 2
        token_sel_second_exp_int_mask = second_expert_mask
        expert_sel_top_c_token_idx = torch.topk(token_sel_first_exp_int_mask + token_sel_second_exp_int_mask,
                                                k=capacity,
                                                dim=0,
                                                sorted=True)[1]
        expert_select_token_idx = expert_sel_top_c_token_idx.t().reshape(num_experts * capacity)
        token_rearranged_ec_idx = torch.cat([token_rearranged_first_ec_idx, token_rearranged_second_ec_idx], dim=0)
        token_exp_weights = torch.cat([token_first_exp_weights, token_second_exp_weights], dim=0)

        top2_gating_token_infos = GatingTokenRearrangeInfo(token_rearranged_ec_idx=token_rearranged_ec_idx,
                                                           token_exp_weights=token_exp_weights,
                                                           expert_select_token_idx=expert_select_token_idx)
        return l_aux, top2_gating_token_infos
    else:
        # Calculate combine_weights and dispatch_mask
        gates1 = einsum("s,se->se", token_first_exp_weights, first_expert_mask_float)
        gates2 = einsum("s,se->se", token_second_exp_weights, second_expert_mask_float)
        locations1_sc = _one_hot_to_float(token_idx_in_first_expert, capacity)
        locations2_sc = _one_hot_to_float(token_idx_in_second_expert, capacity)
        combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
        combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
        combine_weights = combine1_sec + combine2_sec
        dispatch_mask = combine_weights.bool()

        return l_aux, combine_weights, dispatch_mask
