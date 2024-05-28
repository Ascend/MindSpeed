# Copyright (c) Microsoft Corporation.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# copied from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py
# reworked/refactored some parts to make it run.
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.nn import Module

from .utils import _AllToAll, einsum

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self,
                 gate: Module,
                 experts: Module,
                 ep_size,
                 num_local_experts: int, ) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts

    def set_ep_group(self, ep_group):
        self.ep_group = ep_group

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        d_model = input[0].shape[-1]
        reshaped_input = input[0].reshape(-1, d_model)

        # gate
        self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input)
        dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)

        # dispatch all2all
        dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)
        expert_output = self.experts(dispatched_input)

        # combine all2all
        expert_output = _AllToAll.apply(self.ep_group, expert_output)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)

        combined_output = einsum("sec,ecm->sm", combine_weights.type_as(input[0]), expert_output)
        return combined_output.reshape(input[0].shape)