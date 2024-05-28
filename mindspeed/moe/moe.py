# Copyright (c) Microsoft Corporation.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/layer.py
# reworked/refactored some parts to make it run.
import typing

import torch

from .experts import Experts
from .gate import TopKGate
from .moe_layer import MOELayer
from .config import Config


class MoE(torch.nn.Module):
    """Initialize an MoE layer.

    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        aux_loss_coef (int, optional): default=0.0, scaling coefficient for the aux loss.
        z_loss_coef (int, optional): default=0.0, scaling coefficient for the z loss.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
    """

    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 ep_size=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 aux_loss_coef=0.0,
                 z_loss_coef=0.0,
                 ep_group=None,
                 noisy_gate_policy: typing.Optional[str] = None):
        super(MoE, self).__init__()

        assert num_experts % ep_size == 0, f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        num_local_experts = num_experts // ep_size
        config = Config(hidden_size=hidden_size,
                        num_experts=num_experts,
                        ep_size=ep_size,
                        topk=k,
                        capacity_factor=capacity_factor,
                        eval_capacity_factor=eval_capacity_factor,
                        min_capacity=min_capacity,
                        aux_loss_coef=aux_loss_coef,
                        z_loss_coef=z_loss_coef,
                        noisy_gate_policy=noisy_gate_policy
                        )
        self.moe_layer = MOELayer(TopKGate(config),
                                  Experts(expert, num_local_experts),
                                  ep_size,
                                  num_local_experts)
        self.moe_layer.set_ep_group(ep_group)

    def forward(self, hidden_states, used_token=None):
        output = self.moe_layer(hidden_states, used_token)
        return output, self.moe_layer.l_aux, self.moe_layer.exp_counts