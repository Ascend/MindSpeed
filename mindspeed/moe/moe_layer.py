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
import torch.distributed as dist

from .utils import _AllToAll, einsum
from .pipe_experts import PipeExpert

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
                 num_local_experts: int,
                 pipe_experts: bool = False,
                 sequence_parallel: bool = False,
                 pipe_experts_multi_data: int = 1,
                 pipe_experts_multi_stream: bool = False) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts
        self.num_experts = ep_size * num_local_experts
        self.exp_counts = None
        self.l_aux = None

        self.cur_index_window = 0
        self.capacity_window_size = 20
        self.capacity_history_window = []
        self.gate.config.dynamic_capacity = torch.ceil(torch.tensor(256)).to(torch.int64)

        self.pipe_experts = pipe_experts
        self.sequence_parallel = sequence_parallel
        self.pipe_experts_multi_data = pipe_experts_multi_data
        self.pipe_experts_multi_stream = pipe_experts_multi_stream

    def set_ep_group(self, ep_group):
        self.ep_group = ep_group

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        d_model = input[0].shape[-1]
        reshaped_input = input[0].reshape(-1, d_model)
        from megatron.training import get_args
        all_args = get_args()
        # gate
        if not all_args.enable_token_rearrange_opt:
            if self.gate.config.dynamic_padding:
                self.l_aux, combine_weights, dispatch_mask, cur_capacity_cur_rank = self.gate(reshaped_input)
                self.capacity_history_window.append(cur_capacity_cur_rank)
                self.cur_index_window += 1
                if len(self.capacity_history_window) > self.capacity_window_size:
                    self.capacity_history_window.pop(0)
                if self.cur_index_window == self.capacity_window_size - 1:
                    self.cur_index_window = 0
                    capacity_history_window_tensor = torch.Tensor(self.capacity_history_window[-5:]).to(combine_weights.device)
                    dist.all_reduce(capacity_history_window_tensor, op=torch.distributed.ReduceOp.MAX,
                                    group=dist.group.WORLD)
                    self.capacity_history_window = capacity_history_window_tensor.cpu().numpy().tolist()

                    if len(self.capacity_history_window) > 0:
                        capacity_next_window = sum(self.capacity_history_window) / len(self.capacity_history_window) + 20
                    else:
                        capacity_next_window = 256
                    self.gate.config.dynamic_capacity = torch.ceil(torch.tensor(capacity_next_window)).to(torch.int64)
            else:
                self.l_aux, combine_weights, dispatch_mask = self.gate(reshaped_input)
            dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)
        else:
            self.l_aux, token_rearrange_infos = self.gate(reshaped_input)
            org_dtype = reshaped_input.dtype
            if org_dtype == torch.bfloat16:  # 规避算子性能劣化问题, 解决后可删除
                rearranged_input = torch.index_select(
                    reshaped_input.to(torch.float32), dim=0, index=token_rearrange_infos.expert_select_token_idx
                ).to(org_dtype)
            else:
                rearranged_input = torch.index_select(
                    reshaped_input, dim=0, index=token_rearrange_infos.expert_select_token_idx
                )
            capacity = token_rearrange_infos.expert_select_token_idx.size(0) // self.num_experts
            dispatched_input = rearranged_input.reshape(self.num_experts, capacity, d_model).contiguous()

        if self.pipe_experts:
            expert_output = PipeExpert.apply(self.experts, dispatched_input, self.ep_size, self.num_local_experts,
                                             self.sequence_parallel, self.pipe_experts_multi_data,
                                             self.pipe_experts_multi_stream)
        else:
            # dispatch all2all
            dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)

            # Re-shape after all-to-all: ecm -> gecm
            dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)
            expert_output = self.experts(dispatched_input)

            # combine all2all
            expert_output = _AllToAll.apply(self.ep_group, expert_output)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)

        if not all_args.enable_token_rearrange_opt:
            combined_output = einsum("sec,ecm->sm", combine_weights.type_as(input[0]), expert_output)
        else:
            E, C, M = expert_output.shape
            org_dtype = expert_output.dtype
            if org_dtype == torch.bfloat16:
                valid_expert_out = torch.index_select(
                    expert_output.view(E * C, M).to(torch.float32), dim=0, index=token_rearrange_infos.token_rearranged_ec_idx
                ).to(org_dtype)
            else:
                valid_expert_out = torch.index_select(expert_output.view(E * C, M), dim=0, index=token_rearrange_infos.token_rearranged_ec_idx)
            combined_output = valid_expert_out * token_rearrange_infos.token_exp_weights.unsqueeze(1).type_as(input[0])
            if all_args.moe_router_topk == 2:
                combined_output = torch.add(*torch.chunk(combined_output, all_args.moe_router_topk, dim=0))
        return combined_output.reshape(input[0].shape)