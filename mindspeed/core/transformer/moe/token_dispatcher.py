# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.moe.moe_utils import permute, unpermute
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim_expert_parallel
from mindspeed.core.transformer.moe.router import gather_from_sequence_parallel_region_to_moe_async
from mindspeed.core.transformer.moe.comm_utils import async_all_to_all, async_reduce_scatter
from mindspeed.core.transformer.moe.moe_layer_overlap_all2all import forward_func
from mindspeed.core.transformer.moe.unpermute_without_activation import UnpermuteWithoutActivation


def allgather_token_permutation(self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind):
    args = get_args()
    self.hidden_shape = hidden_states.shape
    # [S/TP, B, H] -> [S*B/TP, H]
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

    # Permute the tokens across the expert parallel devices.
    if self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1):
        # [S*B/TP, H] -> [S*B, H]
        with torch.no_grad():
            global_indices, gi_handle = max_ind if isinstance(max_ind, tuple) else gather_from_sequence_parallel_region_to_moe_async(max_ind)
        global_probs, gp_handle = gather_from_sequence_parallel_region_to_moe_async(max_prob)
        global_hidden_states, ghs_handle = gather_from_sequence_parallel_region_to_moe_async(hidden_states)

        with torch.no_grad():
            gi_handle.wait()
            global_local_mask = (global_indices >= self.local_expert_indices[0]) & \
                                (global_indices <= self.local_expert_indices[-1])
            # masked_select
            # local_indices = global_indices.masked_select(global_local_mask)
            # masked_select -> reshape
            local_indices = global_indices.reshape(-1)[global_local_mask.reshape(-1)]
            self.indices = torch.argsort(local_indices.float(), dim=0)
            num_global_experts = self.num_local_experts * parallel_state.get_expert_model_parallel_world_size()
            if args.moe_tp_extend_ep:
                num_global_experts *= parallel_state.get_tensor_model_parallel_world_size()
            all_tokens_per_expert = torch.histc(
                global_indices,
                bins=num_global_experts,
                min=0,
                max=num_global_experts - 1,
            )
        self.all_tokens_per_expert = all_tokens_per_expert.cpu().to(torch.long)
        tokens_per_expert = self.all_tokens_per_expert[self.local_expert_indices[0]: self.local_expert_indices[-1] + 1]
        self.global_local_map = global_local_mask.nonzero()[:, 0]

        if self.router_topk > 1:  # k > 1
            gp_handle.wait()
            # masked_select
            # self.local_probs = global_probs.masked_select(global_local_mask)
            # masked_select -> reshape
            self.local_probs = global_probs.reshape(-1)[global_local_mask.reshape(-1)]
        else:
            self.local_probs = max_prob

        ghs_handle.wait()
        local_hidden_states = global_hidden_states[self.global_local_map, :]
    else:
        if self.router_topk > 1:
            global_local_mask = torch.ones_like(max_ind).bool()
            local_indices = max_ind.masked_select(global_local_mask)
            self.local_probs = max_prob.masked_select(global_local_mask)
            self.global_local_map = global_local_mask.nonzero()[:, 0]
            local_hidden_states = hidden_states[self.global_local_map, :]
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

    permuted_local_hidden_states = local_hidden_states[self.indices, :]
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
    unpermuted_local_hidden = torch.zeros_like(hidden_states)
    unpermuted_local_hidden.index_put_((self.indices,), hidden_states[:self.indices.shape[0], :], accumulate=False)

    # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
    if self.router_topk > 1:
        unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)

    unpermuted_local_bias = None
    if self.add_bias:
        assert bias is not None
        unpermuted_local_bias = torch.zeros_like(hidden_states)
        unpermuted_local_bias.index_put_((self.indices,), bias[:self.indices.shape[0], :], accumulate=False)

        if self.router_topk > 1:
            unpermuted_local_bias = unpermuted_local_bias * scores.view(-1, 1)

    output_total = unpermuted_local_hidden
    output_bias_total = unpermuted_local_bias

    # Unpermute the tokens across expert parallel devices.
    if self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1):
        assert (
                self.global_local_map is not None
        ), "global_local_map is necessary for `AllGather`."
        ep_group_size = parallel_state.get_tensor_and_expert_parallel_world_size()
        # hidden_shape: [SeqLen/TP, MBS, HiddenSize], glboal_num_tokens = SeqLen/TP*MBS*(TP*EP)
        global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * ep_group_size
        global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
        unpermuted_global_hidden = torch.zeros(global_hidden_shape, dtype=torch.float, device=torch.cuda.current_device())
        unpermuted_global_hidden = NewIndePut.apply(unpermuted_global_hidden, (self.global_local_map,),
                                            unpermuted_local_hidden[:self.global_local_map.shape[0], :])

        output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(unpermuted_global_hidden)
        if self.add_bias:
            # Unpermute the bias across expert parallel devices.
            unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
            unpermuted_global_bias.index_put_((self.global_local_map,),
                                              unpermuted_local_bias[:self.global_local_map.shape[0], :],
                                              accumulate=True)

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
            output_total = unpermuted_global_hidden.index_put((self.global_local_map,),
                                                              unpermuted_local_hidden[:self.global_local_map.shape[0], :],
                                                              accumulate=True)
            if self.add_bias:
                unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                output_bias_total = unpermuted_global_bias.index_put((self.global_local_map,),
                                                                     unpermuted_local_bias[:self.global_local_map.shape[0], :],
                                                                     accumulate=True)

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
    self, hidden_states: torch.Tensor, scores: torch.Tensor, indices: torch.Tensor,
):
    self.hidden_shape = hidden_states.shape
    self.scores = scores
    assert scores.dim() == 2, "Expected 2D tensor for scores"
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
    self.local_input_tokens_global_experts_indices = indices
    permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
        hidden_states, self.local_input_tokens_global_experts_indices, topk=self.router_topk,
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
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        global_input_tokens, self.reversed_global_input_permutation_mapping = permute(
            global_input_tokens, self.global_input_tokens_local_experts_indices
        )

    # Perform tensor parallel All-Gather
    # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and self.config.moe_grouped_gemm:
        global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
            global_input_tokens
        )

    return global_input_tokens, tokens_per_expert


def preprocess_tp_extend_ep(self, indices: torch.Tensor) -> torch.Tensor:
    num_local_tokens_per_expert = torch.histc(
        indices, bins=self.num_experts, min=0, max=self.num_experts
    )
    # num_local_tokens_per_expert: [num_experts]

    ep_size = self.config.expert_model_parallel_size
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_extended_ep_size = ep_size * tp_size
    if tp_extended_ep_size > 1:
        # ===================================================
        # Calculate input_splits, output_splits for alltoall-v.
        # ===================================================
        self.input_splits = (
            num_local_tokens_per_expert.reshape(tp_extended_ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"))
            .numpy()
        )
        num_global_tokens_per_expert = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
            num_local_tokens_per_expert
        ).reshape(tp_extended_ep_size, self.num_experts)
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


def alltoall_token_permutation_tp_extend_ep(
    self, hidden_states: torch.Tensor, scores: torch.Tensor, indices: torch.Tensor,
):
    self.hidden_shape = hidden_states.shape
    self.scores = scores
    assert scores.dim() == 2, "Expected 2D tensor for scores"
    assert indices.dim() == 2, "Expected 2D tensor for indices"
    tokens_per_expert = self.preprocess(indices)

    # Flatten the input tensor
    # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

    # Permutation 1: input to AlltoAll input
    self.local_input_tokens_global_experts_indices = indices
    permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
        hidden_states, self.local_input_tokens_global_experts_indices, topk=self.router_topk,
    )

    # Perform expert parallel AlltoAll communication
    global_input_tokens = tensor_parallel.all_to_all(
        parallel_state.get_tensor_and_expert_parallel_group(),
        permutated_local_input_tokens,
        self.output_splits,
        self.input_splits,
    )

    # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
    if self.num_local_experts > 1:
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        global_input_tokens, self.reversed_global_input_permutation_mapping = permute(
            global_input_tokens, self.global_input_tokens_local_experts_indices
        )


    return global_input_tokens, tokens_per_expert


def alltoall_token_unpermutation_tp_extend_ep(
    self, hidden_states: torch.Tensor, bias: torch.Tensor = None,
):
    """
    Reverse the token permutation to restore the original order.

    Args:
        hidden_states (torch.Tensor): Output from local experts.
        bias (torch.Tensor, optional): Bias tensor (not supported).

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - Unpermuted token embeddings in the original order.
            - None (bias is not supported).
    """
    assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

    # Unpermutation 2: expert output to AlltoAll input
    # hidden_states: [SEQL, H] -> [SEQL, H/TP]
    if self.num_local_experts > 1:
        hidden_states = unpermute(
            hidden_states, self.reversed_global_input_permutation_mapping,
        )

    # Perform expert parallel AlltoAll communication
    permutated_local_input_tokens = tensor_parallel.all_to_all(
        parallel_state.get_tensor_and_expert_parallel_group(),
        hidden_states,
        self.input_splits,
        self.output_splits,
    )

    # Unpermutation 1: AlltoAll output to output
    output = unpermute(
        permutated_local_input_tokens,
        self.reversed_local_input_permutation_mapping,
        probs=self.scores,
        topk=self.router_topk,
    )

    # Reshape the output tensor
    output = output.view(self.hidden_shape)
    return output, None


def allgather_token_permutation_new(self, global_indices_2_tuple, global_probs_2_tuple, global_hidden_states_2_tuple):
    global_indices, gi_handle = global_indices_2_tuple
    global_probs, gp_handle = global_probs_2_tuple
    global_hidden_states, ghs_handle = global_hidden_states_2_tuple

    local_hidden_states = None
    tokens_per_expert = None

    if self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1):
        with torch.no_grad():
            gi_handle.wait()
            global_local_mask = (global_indices >= self.local_expert_indices[0]) & (global_indices <= self.local_expert_indices[-1])

            # masked_select -> reshape
            local_indices = global_indices.reshape(-1)[global_local_mask.reshape(-1)]
            self.indices = torch.argsort(local_indices.float(), dim=0)
            num_global_experts = self.num_local_experts * parallel_state.get_expert_model_parallel_world_size()
            if get_args().moe_tp_extend_ep:
                num_global_experts *= parallel_state.get_tensor_model_parallel_world_size()
            all_tokens_per_expert = torch.histc(
                global_indices,
                bins=num_global_experts,
                min=0,
                max=num_global_experts
            )
        self.all_tokens_per_expert = all_tokens_per_expert.cpu().to(torch.long)
        tokens_per_expert = self.all_tokens_per_expert[self.local_expert_indices[0]: self.local_expert_indices[-1] + 1]
        self.global_local_map = global_local_mask.nonzero()[:, 0]

        if self.router_topk > 1:  # k > 1
            gp_handle.wait()
            # masked_select -> reshape
            self.local_probs = global_probs.reshape(-1)[global_local_mask.reshape(-1)]

        ghs_handle.wait()
        local_hidden_states = global_hidden_states[self.global_local_map, :]

    permuted_local_hidden_states = local_hidden_states[self.indices, :]
    return (
        permuted_local_hidden_states,
        tokens_per_expert,
        self.global_local_map,
        self.indices
    )


def allgather_token_unpermutation_new(self, hidden_states: torch.Tensor, bias: torch.Tensor = None):
    # Stage1: unpermute the tokens and bias locally respectively.w
    scores = self.local_probs.to(dtype=hidden_states.dtype)
    unpermuted_local_hidden = torch.zeros_like(hidden_states)
    unpermuted_local_hidden.index_put_((self.indices,), hidden_states[:self.indices.shape[0], :], accumulate=False)

    # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
    if self.router_topk > 1:
        unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)

    unpermuted_local_bias = None
    if self.add_bias:
        assert bias is not None
        unpermuted_local_bias = torch.zeros_like(hidden_states)
        unpermuted_local_bias.index_put_((self.indices,), bias[:self.indices.shape[0], :], accumulate=False)

        if self.router_topk > 1:
            unpermuted_local_bias = unpermuted_local_bias * scores.view(-1, 1)

    output_total = unpermuted_local_hidden
    output_bias_total = unpermuted_local_bias

    # Unpermute the tokens across expert parallel devices.
    if self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1):
        assert (
            self.global_local_map is not None
        ), "global_local_map is necessary for 'AllGather'."
        ep_group_size = parallel_state.get_tensor_and_expert_parallel_world_size()
        # hidden_shape: [SeqLen/TP, MBS, HiddenSize], global_num_tokens = SeqLen/TP*MBS*(TP*EP)
        global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * ep_group_size
        global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]

        unpermuted_global_hidden = torch.zeros(global_hidden_shape, dtype=torch.float, device=torch.cuda.current_device())
        unpermuted_global_hidden = NewIndePut.apply(unpermuted_global_hidden, (self.global_local_map,),
                                            unpermuted_local_hidden[:self.global_local_map.shape[0], :])
        output_total = unpermuted_global_hidden
        if self.add_bias:
            # Unpermute the bias across expert parallel devices.
            unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
            unpermuted_global_bias.index_put_((self.global_local_map,),
                                              unpermuted_local_bias[:self.global_local_map.shape[0], :],
                                              accumulate=True)

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
                device=torch.cuda.current_device()
            )
            output_total = unpermuted_global_hidden.index_put((self.global_local_map,),
                                                              unpermuted_local_hidden[:self.global_local_map.shape[0], :],
                                                              accumulate=True)

            if self.add_bias:
                unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                output_bias_total = unpermuted_global_bias.index_put((self.global_local_map,),
                                                                     unpermuted_local_bias[:self.global_local_map.shape[0], :],
                                                                     accumulate=True)

    if self.router_topk == 1:
        output_total = output_total * scores
    if self.add_bias:
        assert output_bias_total is not None
        if self.router_topk == 1:
            output_bias_total = output_bias_total * scores
        output_bias_total = output_bias_total.view(self.hidden_shape)
    else:
        output_bias_total = None

    return output_total, output_bias_total


def alltoall_token_permutation_new(
        self, hidden_states: torch.Tensor, scores: torch.Tensor, indices: torch.Tensor, shared_experts, save_tensors, save_tensors_for_grad, moe_ctx=None
):
    self.hidden_shape = hidden_states.shape
    self.scores = scores
    assert scores.dim() == 2, "Expected 2D tensor for scores"
    assert indices.dim() == 2, "Expected 2D tensor for indices"

    def alltoall_token_permutation1(hidden_states, indices):
        tokens_per_expert = self.preprocess(indices)

        # Flatten the input tensor
        # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Perform tensor parallel AlltoAll communication
        # hidden_states: [S*B/TP, H] -> [S*B, H/TP]
        if not get_args().moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)

        # Permutation 1: input to AlltoAll input
        self.local_input_tokens_global_experts_indices = indices
        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
            hidden_states, self.local_input_tokens_global_experts_indices, topk=self.router_topk
        )
        return tokens_per_expert, permutated_local_input_tokens

    (tokens_per_expert, permutated_local_input_tokens), *_ = forward_func(alltoall_token_permutation1,
                                                                          (hidden_states, indices))

    # permute 1
    save_tensors.append(permutated_local_input_tokens)

    # Perform expert parallel AlltoAll communication
    ep_group = parallel_state.get_expert_model_parallel_group()
    if get_args().moe_tp_extend_ep:
        ep_group = parallel_state.get_tensor_and_expert_parallel_group()
    _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
        permutated_local_input_tokens,
        self.output_splits,
        self.input_splits,
        ep_group,
    )

    # shared experts
    if shared_experts is not None:
        (share_experts_output, _), *_ = forward_func(shared_experts, (hidden_states, moe_ctx))
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            share_experts_graph, share_experts_output, rs_shared_experts_handle = async_reduce_scatter(share_experts_output, parallel_state.get_tensor_model_parallel_group(),
                                                                                                       event=permute1_ep_all_to_all_handle, stream=torch.npu.default_stream())
            share_experts_output = (share_experts_graph, share_experts_output, rs_shared_experts_handle)
    else:
        share_experts_output = None

    permute1_ep_all_to_all_handle.wait()
    permutated_local_input_tokens.untyped_storage().resize_(0)
    # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
    if self.num_local_experts > 1:
        if self.comm_stream is not None:
            torch.cuda.current_stream().wait_stream(self.comm_stream)

        def alltoall_token_permutation2(global_input_tokens):
            global_input_tokens, self.reversed_global_input_permutation_mapping = permute(
                global_input_tokens, self.global_input_tokens_local_experts_indices
            )
            # Perform tensor parallel All-Gather
            # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
            if (not get_args().moe_tp_extend_ep and
                parallel_state.get_tensor_model_parallel_world_size() > 1 and
                self.config.moe_grouped_gemm):
                global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
                    global_input_tokens
                )
            return global_input_tokens

        # token 重排2 input
        (global_input_tokens), global_input_tokens_detach = forward_func(alltoall_token_permutation2,
                                                                         global_input_tokens)
        save_tensors.append(global_input_tokens_detach)
        save_tensors.append(global_input_tokens)
        save_tensors_for_grad.append(global_input_tokens_detach)
        global_input_tokens_detach.untyped_storage().resize_(0)

    return share_experts_output, global_input_tokens, tokens_per_expert


def alltoall_token_unpermutation_new(
        self, hidden_states, bias, save_tensors, save_tensors_for_grad
):
    def alltoall_token_unpermutation1(hidden_states):
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        # Perform tensor parallel Reduce-Scatter
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        if not get_args().moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.reduce_scatter_last_dim_to_tensor_parallel_region(hidden_states)

        # Unpermutation 2: expert output to AlltoAll input
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        if self.num_local_experts > 1:
            hidden_states = unpermute(
                hidden_states, self.reversed_global_input_permutation_mapping
            )
        return hidden_states

    hidden_states, unpermute1_input_detach = forward_func(alltoall_token_unpermutation1, hidden_states)
    save_tensors.append(unpermute1_input_detach)
    save_tensors.append(hidden_states)
    save_tensors_for_grad.append(unpermute1_input_detach)
    unpermute1_input_detach.untyped_storage().resize_(0)

    ep_group = parallel_state.get_expert_model_parallel_group()
    if get_args().moe_tp_extend_ep:
        ep_group = parallel_state.get_tensor_and_expert_parallel_group()
    _, permutated_local_input_tokens, handle = async_all_to_all(
        hidden_states,
        self.input_splits,
        self.output_splits,
        ep_group
    )
    handle.wait()
    hidden_states.untyped_storage().resize_(0)

    def alltoall_token_unpermutation2(permutated_local_input_tokens):
        # Unpermutation 1: AlltoAll output to output
        if get_args().moe_without_activation:
            output = UnpermuteWithoutActivation.apply(
                permutated_local_input_tokens,
                self.reversed_local_input_permutation_mapping,
                self.scores
            )
        else:
            output = unpermute(
                permutated_local_input_tokens,
                self.reversed_local_input_permutation_mapping,
                probs=self.scores,
                topk=self.router_topk
            )

        # Perform tensor parallel AlltoAll communication
        if not get_args().moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
            # output: [S*B, H/TP] -> [S*B/TP, H]
            output = tensor_parallel.all_to_all_hp2sp(output)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output

    output, unpermute2_input_detach = forward_func(alltoall_token_unpermutation2, permutated_local_input_tokens)
    save_tensors.append(unpermute2_input_detach)
    save_tensors_for_grad.append(unpermute2_input_detach)
    if not get_args().use_fused_moe_token_permute_and_unpermute or get_args().moe_without_activation:
        unpermute2_input_detach.untyped_storage().resize_(0)
    return output, None
