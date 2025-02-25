# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import List

import torch

from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_utils import moe_gather
from megatron.core.transformer.moe.moe_utils import moe_scatter
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from mindspeed.core.tensor_parallel.comm_autograd_function import \
    auto_grad_reduce_scatter_along_first_dim
from mindspeed.core.tensor_parallel.comm_autograd_function import \
    auto_grad_sync_gather_along_first_dim_rs
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm
from mindspeed.core.tensor_parallel.comm_group_api import TPXEPCollectiveComm
from mindspeed.core.transformer.moe.token_dispatcher import NewIndePut
from mindspeed.core.transformer.moe.token_dispatcher import cann_version_check


class MoEAllGatherTokenDispatcher2D(MoETokenDispatcher):
    """
    AllGather Based Token dispatcher.
    """

    def __init__(
            self, num_local_experts: int, local_expert_indices: List[int], config: TransformerConfig,
    ) -> None:
        """
        Initialize the zero token dropping router.
        """
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert len(self.local_expert_indices) > 0, "Expected at least one local expert index"
        self.router_topk = config.moe_router_topk
        self.add_bias = config.add_bias_linear

        # self.local_probs: probs of global token assignment to local experts.
        self.local_probs = None

        # self.indices: The indices of `local_indices`
        self.indices = None

        # self.global_local_map: 2D tensor
        self.global_local_map = None

    def token_permutation(
            self, hidden_states: torch.Tensor, topk_probs: torch.Tensor, topk_indices: torch.Tensor
    ):
        """Dispatch tokens to local experts. It's composed of two stages:
        (1) Permute the tokens across the expert parallel devices. After this stage,
        each device receives all the tokens assigned to its local set of experts
        in its local HBM.
        (2) Permute the tokens locally so that they are grouped by their expert
        assignment.
         After the stage (1), the tokens are grouped by which device
        they came from. We re-order them locally for subsequent efficient computation.

        Args:
            hidden_states: input tokens of shape [s/(cp*x), b, h]
            topk_probs: probs of local token assignment to global experts
            with shape: [sb/(cp*x), topK]
            topk_indices: token assignment to local experts with shape: [sb/(cp*x), topK]

        Returns:
            permuted_local_hidden_states: Permutation of tokens to local experts group.
            tokens_per_expert: the number of tokens each local expert to process.
        """

        self.hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/(cp*x), H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Permute the tokens across the expert parallel devices.
        if TPXCollectiveComm.get_comm_group_world_size() > 1 or self.config.expert_model_parallel_size > 1:
            # [S*B/(cp*x), H] -> [S*B, H]
            with torch.no_grad():
                # [sb/x, topk] -> [sb*ep, topK]
                global_indices = auto_grad_sync_gather_along_first_dim_rs(topk_indices, TPXEPCollectiveComm)

            # [sb/x, topk] -> [sb*ep, topK]
            global_probs = auto_grad_sync_gather_along_first_dim_rs(topk_probs, TPXEPCollectiveComm)
            # [S/x, b, h] -> [sb*ep, h]
            global_hidden_states = auto_grad_sync_gather_along_first_dim_rs(hidden_states, TPXEPCollectiveComm)

            with torch.no_grad():
                global_local_mask = (global_indices >= self.local_expert_indices[0]) & (
                            global_indices <= self.local_expert_indices[-1])
                local_indices = global_indices.masked_select(global_local_mask)
                self.indices = torch.argsort(local_indices.float(), dim=0)
                num_global_experts = self.num_local_experts * parallel_state.get_expert_model_parallel_world_size()

                all_tokens_per_expert = torch.histc(global_indices, bins=num_global_experts, min=0,
                    max=num_global_experts - 1, )
            self.all_tokens_per_expert = all_tokens_per_expert.to(torch.long)
            tokens_per_expert = self.all_tokens_per_expert[
                                self.local_expert_indices[0]: self.local_expert_indices[-1] + 1]
            self.global_local_map = global_local_mask.nonzero()[:, 0]

            if self.router_topk > 1:
                self.local_probs = global_probs.masked_select(global_local_mask)
            else:
                self.local_probs = topk_probs

            if cann_version_check:
                local_hidden_states = global_hidden_states[self.global_local_map, :]
            else:
                self.global_local_map = (self.global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1]))
                local_hidden_states = moe_gather.apply(global_hidden_states, self.global_local_map)
        else:
            if self.router_topk > 1:
                global_local_mask = torch.ones_like(topk_indices).bool()
                local_indices = topk_indices.masked_select(global_local_mask)
                self.local_probs = topk_probs.masked_select(global_local_mask)
                self.global_local_map = global_local_mask.nonzero()[:, 0]
                if cann_version_check:
                    local_hidden_states = hidden_states[self.global_local_map, :]
                else:
                    self.global_local_map = self.global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
                    local_hidden_states = torch.gather(hidden_states, 0, self.global_local_map)
            else:
                local_indices = topk_indices
                self.local_probs = topk_probs
                local_hidden_states = hidden_states
                self.global_local_map = None

            with torch.no_grad():
                # The indices of local_indices that give its sorted order along dim 0.
                self.indices = torch.argsort(local_indices, dim=0)
                # use 0.7.0 implement for better performance
                tokens_per_expert = torch.histc(local_indices, bins=self.num_local_experts,
                    min=self.local_expert_indices[0], max=self.local_expert_indices[-1], )
                tokens_per_expert = tokens_per_expert.to(torch.long)
            self.all_tokens_per_expert = tokens_per_expert

        if self.num_local_experts > 1:
            if cann_version_check:
                permuted_local_hidden_states = local_hidden_states[self.indices, :]
            else:
                self.indices = self.indices.view(-1, 1).expand(-1, hidden_states.shape[-1])
                permuted_local_hidden_states = moe_gather.apply(local_hidden_states, self.indices)
        else:
            permuted_local_hidden_states = local_hidden_states

        return permuted_local_hidden_states, tokens_per_expert


    def token_unpermutation(
            self,
            hidden_states: torch.Tensor,
            bias: torch.Tensor = None,
    ):
        """
        Reverse process of `dispatch()` which permutes the output of local
        experts locally and across expert parallel rank into the original order to
        produce the final output.

        Args:
            hidden_states: 2D tensor of shape [sum_tokens_of_all_local_experts, HiddenSize],
            output of local experts.
            bias (optional): The bias tensor.

        Returns:
            output_total: un-permuted updated hidden states output from all local experts
            with shape of [SeqLen/TP, MBS, HiddenSize]
        """
        # Stage1: unpermute the tokens and bias locally respectively.
        scores = self.local_probs.to(dtype=hidden_states.dtype)
        if self.num_local_experts > 1:
            if cann_version_check:
                unpermuted_local_hidden = torch.zeros_like(hidden_states)
                unpermuted_local_hidden.index_put_((self.indices,),
                                                   hidden_states[:self.indices.shape[0], :],
                                                   accumulate=False)
            else:
                assert self.indices.shape == hidden_states.shape
                unpermuted_local_hidden = moe_scatter.apply(hidden_states, self.indices)
        else:
            unpermuted_local_hidden = hidden_states

        # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
        if self.router_topk > 1:
            unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)

        unpermuted_local_bias = None
        if self.add_bias:
            assert bias is not None
            unpermuted_local_bias = torch.zeros_like(hidden_states)
            if cann_version_check:
                unpermuted_local_bias.index_put_((self.indices,), bias[:self.indices.shape[0], :],
                                                 accumulate=False)
            else:
                assert self.indices.shape == bias.shape
                unpermuted_local_bias = unpermuted_local_bias.scatter(0, self.indices, bias)
            if self.router_topk > 1:
                unpermuted_local_bias = unpermuted_local_bias * scores.view(-1, 1)

        output_total = unpermuted_local_hidden
        output_bias_total = unpermuted_local_bias

        # Unpermute the tokens across expert parallel devices.
        if TPXCollectiveComm.get_comm_group_world_size() > 1 or self.config.expert_model_parallel_size > 1:
            assert (self.global_local_map is not None), \
                "global_local_map is necessary for `AllGather`."
            ep_group_size = TPXEPCollectiveComm.get_comm_group_world_size()
            # hidden_shape: [SeqLen/TP, MBS, HiddenSize], glboal_num_tokens = SeqLen/TP*MBS*(TP*EP)
            global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * ep_group_size
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
            if cann_version_check:
                unpermuted_global_hidden = torch.zeros(global_hidden_shape, dtype=torch.float,
                                                       device=torch.cuda.current_device())
                unpermuted_global_hidden = NewIndePut.apply(unpermuted_global_hidden,
                                                            (self.global_local_map,),
                                                            unpermuted_local_hidden[
                                                            :self.global_local_map.shape[0], :])
            else:
                assert self.global_local_map.shape == unpermuted_local_hidden.shape
                unpermuted_global_hidden = moe_scatter.apply(unpermuted_local_hidden,
                    self.global_local_map, global_hidden_shape)

            output_total = auto_grad_reduce_scatter_along_first_dim(unpermuted_global_hidden, TPXEPCollectiveComm)
            if self.add_bias:
                # Unpermute the bias across expert parallel devices.
                unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                if cann_version_check:
                    unpermuted_global_bias.index_put_((self.global_local_map,),
                                                      unpermuted_local_bias[
                                                      :self.global_local_map.shape[0], :],
                                                      accumulate=True)
                else:
                    unpermuted_global_bias = unpermuted_global_bias.scatter_add(0,
                        self.global_local_map, unpermuted_local_bias)

                output_bias_total = auto_grad_reduce_scatter_along_first_dim(unpermuted_global_bias,
                                                                             TPXEPCollectiveComm)
                # bias is duplicated across tensor parallelism ranks;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = (output_bias_total /
                                     TPXCollectiveComm.get_comm_group_world_size())
        else:
            if self.router_topk > 1:
                global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1]
                global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
                unpermuted_global_hidden = torch.zeros(global_hidden_shape,
                    dtype=hidden_states.dtype, device=torch.cuda.current_device(), )
                if cann_version_check:
                    output_total = unpermuted_global_hidden.index_put((self.global_local_map,),
                                                                      unpermuted_local_hidden[
                                                                      :self.global_local_map.shape[
                                                                          0], :], accumulate=True)
                else:
                    output_total = unpermuted_global_hidden.scatter_add(0, self.global_local_map,
                        unpermuted_local_hidden)
                if self.add_bias:
                    unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                    if cann_version_check:
                        output_bias_total = unpermuted_global_bias.index_put(
                            (self.global_local_map,),
                            unpermuted_local_bias[:self.global_local_map.shape[0], :],
                            accumulate=True)
                    else:
                        output_bias_total = unpermuted_global_bias.scatter_add(0,
                            self.global_local_map, unpermuted_local_bias)

        if self.router_topk == 1:
            output_total = output_total * scores
        output_total = output_total.view(self.hidden_shape)
        if self.add_bias:
            assert output_bias_total is not None
            if self.router_topk == 1:
                output_bias_total = output_bias_total * scores
            output_bias_total = output_bias_total.view(self.hidden_shape)
        else:
            output_bias_total = None

        return output_total, output_bias_total
