# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from torch_npu.utils.collect_env import get_cann_version
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.moe.moe_utils import moe_gather, moe_scatter, permute
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim_expert_parallel
from mindspeed.core.transformer.moe.router import gather_from_sequence_parallel_region_to_moe_async



def is_less_or_equal_rc2_cann_version():
    cann_starts_with = ('8.0.RC1', '8.0.RC2')
    cann_all = ('not known', '8.0.T1', '8.0.T2', '8.0.T3', '8.0.T37', '8.0.T5', '8.0.T6', '8.0.T7',
                '8.0.T8', '8.0.T10', '8.0.T13', '8.0.T16', '8.0.T50', '8.0.T51', '8.0.T52')
    cann_version = get_cann_version()
    return cann_version in cann_all or cann_version.startswith(cann_starts_with)


cann_version_check = is_less_or_equal_rc2_cann_version()


def allgather_token_permutation(self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind):
    self.hidden_shape = hidden_states.shape
    # [S/TP, B, H] -> [S*B/TP, H]
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

    # Permute the tokens across the expert parallel devices.
    if (self.config.tensor_model_parallel_size > 1) or (
        self.config.expert_model_parallel_size > 1
    ):
        # [S*B/TP, H] -> [S*B, H]
        with torch.no_grad():
            global_indices, gi_handle = max_ind if isinstance(max_ind, tuple) else gather_from_sequence_parallel_region_to_moe_async(max_ind)
        global_probs, gp_handle = gather_from_sequence_parallel_region_to_moe_async(max_prob)
        global_hidden_states, ghs_handle = gather_from_sequence_parallel_region_to_moe_async(hidden_states)

        with torch.no_grad():
            gi_handle.wait()
            global_local_mask = (global_indices >= self.local_expert_indices[0]) & \
                                (global_indices <= self.local_expert_indices[-1])
            local_indices = global_indices.masked_select(global_local_mask)
            self.indices = torch.argsort(local_indices, dim=0)
            all_tokens_per_expert = torch.histc(
                global_indices,
                bins=self.num_local_experts * parallel_state.get_expert_model_parallel_world_size(),
                min=0,
                max=self.num_local_experts * parallel_state.get_expert_model_parallel_world_size() - 1,
            )
        self.all_tokens_per_expert = all_tokens_per_expert.cpu().to(torch.long)
        tokens_per_expert = self.all_tokens_per_expert[self.local_expert_indices[0]: self.local_expert_indices[-1] + 1]
        self.global_local_map = global_local_mask.nonzero()[:, 0]

        if self.router_topk > 1:  # k > 1
            gp_handle.wait()
            self.local_probs = global_probs.masked_select(global_local_mask)
        else:
            self.local_probs = max_prob

        ghs_handle.wait()
        if cann_version_check:
            local_hidden_states = global_hidden_states[self.global_local_map, :]
        else:
            self.global_local_map = self.global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
            local_hidden_states = moe_gather.apply(global_hidden_states, self.global_local_map)
    else:
        if self.router_topk > 1:
            global_local_mask = torch.ones_like(max_ind).bool()
            local_indices = max_ind.masked_select(global_local_mask)
            self.local_probs = max_prob.masked_select(global_local_mask)
            self.global_local_map = global_local_mask.nonzero()[:, 0]
            if cann_version_check:
                local_hidden_states = hidden_states[self.global_local_map, :]
            else:
                self.global_local_map = self.global_local_map.view(-1, 1).expand(
                    -1, hidden_states.shape[-1]
                )
                local_hidden_states = torch.gather(hidden_states, 0, self.global_local_map)
        else:
            local_indices = max_ind
            self.local_probs = max_prob
            local_hidden_states = hidden_states
            self.global_local_map = None

        with torch.no_grad():
            # The indices of local_indices that give its sorted order along dim 0.
            self.indices = torch.argsort(local_indices, dim=0)
            tokens_per_expert = torch.histc(
                local_indices,
                bins=self.num_local_experts,
                min=self.local_expert_indices[0],
                max=self.local_expert_indices[-1],
            )
            tokens_per_expert = tokens_per_expert.cpu().to(torch.long)
        self.all_tokens_per_expert = tokens_per_expert

    if self.num_local_experts > 1:
        if cann_version_check:
            permuted_local_hidden_states = local_hidden_states[self.indices, :]
        else:
            self.indices = self.indices.view(-1, 1).expand(-1, hidden_states.shape[-1])
            permuted_local_hidden_states = moe_gather.apply(local_hidden_states, self.indices)
    else:
        permuted_local_hidden_states = local_hidden_states
    return (
        permuted_local_hidden_states,
        tokens_per_expert,
    )


class NewIndePut(torch.autograd.Function):
    @staticmethod
    def forward(self, tensor, map_, value_):
        self.map_ = map_
        ori_dtype = None
        if value_.dtype != torch.float32:
            ori_dtype = value_.dtype
            value_ = value_.float()
        output = tensor.index_put_(map_, value_, accumulate=True)
        if ori_dtype:
            return output.to(ori_dtype)
        return output

    def backward(self, grad_input):
        map_ = self.map_
        grad_output = grad_input.index_select(0, map_[0])
        return None, None, grad_output


def allgather_token_unpermutation(self, hidden_states: torch.Tensor, bias: torch.Tensor = None, ):
    # Stage1: unpermute the tokens and bias locally respectively.w
    scores = self.local_probs.to(dtype=hidden_states.dtype)
    if self.num_local_experts > 1:
        if cann_version_check:
            unpermuted_local_hidden = torch.zeros_like(hidden_states)
            unpermuted_local_hidden.index_put_((self.indices,), hidden_states[:self.indices.shape[0], :], accumulate=False)
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
            unpermuted_local_bias.index_put_((self.indices,), bias[:self.indices.shape[0], :], accumulate=False)
        else:
            assert self.indices.shape == bias.shape
            unpermuted_local_bias = unpermuted_local_bias.scatter(0, self.indices, bias)
        if self.router_topk > 1:
            unpermuted_local_bias = unpermuted_local_bias * scores.view(-1, 1)

    output_total = unpermuted_local_hidden
    output_bias_total = unpermuted_local_bias

    # Unpermute the tokens across expert parallel devices.
    if (self.config.tensor_model_parallel_size > 1) or (
            self.config.expert_model_parallel_size > 1
    ):
        assert (
                self.global_local_map is not None
        ), "global_local_map is necessary for `AllGather`."
        ep_group_size = parallel_state.get_tensor_and_expert_parallel_world_size()
        # hidden_shape: [SeqLen/TP, MBS, HiddenSize], glboal_num_tokens = SeqLen/TP*MBS*(TP*EP)
        global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * ep_group_size
        global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
        if cann_version_check:
            unpermuted_global_hidden = torch.zeros(global_hidden_shape, dtype=torch.float, device=torch.cuda.current_device())
            unpermuted_global_hidden = NewIndePut.apply(unpermuted_global_hidden, (self.global_local_map,),
                                                unpermuted_local_hidden[:self.global_local_map.shape[0], :])
        else:
            assert self.global_local_map.shape == unpermuted_local_hidden.shape
            unpermuted_global_hidden = moe_scatter.apply(
                unpermuted_local_hidden, self.global_local_map, global_hidden_shape
            )

        output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(unpermuted_global_hidden)
        if self.add_bias:
            # Unpermute the bias across expert parallel devices.
            unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
            if cann_version_check:
                unpermuted_global_bias.index_put_((self.global_local_map,),
                                                  unpermuted_local_bias[:self.global_local_map.shape[0], :],
                                                  accumulate=True)
            else:
                unpermuted_global_bias = unpermuted_global_bias.scatter_add(
                    0, self.global_local_map, unpermuted_local_bias
                )

            output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                unpermuted_global_bias
            )
            # bias is duplicated across tensor parallelism ranks;
            # reduce scatter reduces bias across tensor parallel_ranks
            output_bias_total = (output_bias_total / parallel_state.get_tensor_model_parallel_world_size())
    else:
        if self.router_topk > 1:
            global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1]
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
            unpermuted_global_hidden = torch.zeros(
                global_hidden_shape,
                dtype=hidden_states.dtype,
                device=torch.cuda.current_device(),
            )
            if cann_version_check:
                output_total = unpermuted_global_hidden.index_put((self.global_local_map,),
                                                                  unpermuted_local_hidden[:self.global_local_map.shape[0], :],
                                                                  accumulate=True)
            else:
                output_total = unpermuted_global_hidden.scatter_add(
                    0, self.global_local_map, unpermuted_local_hidden
                )
            if self.add_bias:
                unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                if cann_version_check:
                    output_bias_total = unpermuted_global_bias.index_put((self.global_local_map,),
                                                                         unpermuted_local_bias[:self.global_local_map.shape[0], :],
                                                                         accumulate=True)
                else:
                    output_bias_total = unpermuted_global_bias.scatter_add(
                        0, self.global_local_map, unpermuted_local_bias
                    )

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


def preprocess(self, indices: torch.Tensor) -> torch.Tensor:
    num_local_tokens_per_expert = torch.histc(
        indices, bins=self.num_experts, min=0, max=self.num_experts
    )
    # num_local_tokens_per_expert: [num_experts]

    ep_size = self.config.expert_model_parallel_size
    if self.drop_and_pad:
        # probs: [num_experts, capacity]
        self.capacity = self.probs.size(1)
        num_tokens_per_local_expert = torch.full(
            (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long
        )
        return num_tokens_per_local_expert
    elif self.config.moe_expert_capacity_factor is not None:
        self.num_out_tokens = num_local_tokens_per_expert.sum().cpu()

    if ep_size > 1:
        # ===================================================
        # Calculate input_splits, output_splits for alltoall-v.
        # ===================================================
        self.input_splits = (
            num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"))
            .numpy()
        )
        num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(
            num_local_tokens_per_expert
        ).reshape(ep_size, self.num_experts)
        self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices
        ]
        self.output_splits = (
            self.num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu")).numpy()
        )
        num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0).to(
            torch.device("cpu"), non_blocking=True
        )
        # ===================================================
        # num_global_tokens_per_expert: [ep_size, num_experts]
        # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
        # num_tokens_per_local_expert: [num_local_experts]
        # ===================================================
    else:
        self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
            -1, self.num_experts
        )
        num_tokens_per_local_expert = num_local_tokens_per_expert.to(
            torch.device("cpu"), non_blocking=True
        )

    if self.num_local_experts > 1:
        if not hasattr(self, 'comm_stream'):
            self.comm_stream = torch.cuda.Stream()
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.comm_stream):
            expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.config.num_moe_experts)],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
            )

    return num_tokens_per_local_expert


def alltoall_token_permutation(
    self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor,
):
    self.hidden_shape = hidden_states.shape
    self.probs = probs
    assert probs.dim() == 2, "Expected 2D tensor for probs"
    assert indices.dim() == 2, "Expected 2D tensor for indices"
    tokens_per_expert = self.preprocess(indices)

    # Flatten the input tensor
    # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

    # Perform tensor parallel AlltoAll communication
    # hidden_states: [S*B/TP, H] -> [S*B, H/TP]
    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)
    
    # Permutation 1: input to AlltoAll input
    self.hiddden_shape_before_permute = hidden_states.shape
    permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
        hidden_states,
        indices,
        num_out_tokens=self.num_out_tokens,
        padded_mode=self.drop_and_pad,
    )

    # Perform expert parallel AlltoAll communication
    global_input_tokens = tensor_parallel.all_to_all(
        parallel_state.get_expert_model_parallel_group(),
        permutated_local_input_tokens,
        self.output_splits,
        self.input_splits,
    )

    # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
    if self.num_local_experts > 1:
        if not self.drop_and_pad:
            torch.cuda.current_stream().wait_stream(self.comm_stream)
            global_input_tokens, self.reversed_global_input_permutation_mapping = permute(
                global_input_tokens, self.global_input_tokens_local_experts_indices
            )
        else:
            global_input_tokens = global_input_tokens.reshape(
                self.ep_size, self.num_local_experts, self.capacity, -1
            )
            global_input_tokens = (
                global_input_tokens.transpose(0, 1)
                .reshape(self.num_local_experts * self.ep_size * self.capacity, -1)
                .contiguous()
            )

    # Perform tensor parallel All-Gather on the hidden dimension to obtain the input tokens.
    # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and self.config.moe_grouped_gemm:
        global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
            global_input_tokens
        )

    return global_input_tokens, tokens_per_expert