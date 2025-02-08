# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from torch_npu.utils.collect_env import get_cann_version
from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel, mpu
from megatron.core.transformer.moe.moe_utils import moe_gather, moe_scatter, permute, unpermute
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim_expert_parallel, reduce_scatter_to_sequence_parallel_region
from mindspeed.core.transformer.moe.router import gather_from_sequence_parallel_region_to_moe_async
from mindspeed.core.transformer.moe.comm_utils import (async_all_to_all, async_reduce_scatter,
                                                       async_all_gather, reduce_scatter, all_gather)
from mindspeed.core.transformer.moe.moe_layer_overlap_all2all import forward_func
from mindspeed.core.transformer.moe.unpermute_without_activation import UnpermuteWithoutActivation
from mindspeed.core.transformer.moe.moe_utils import (AG_SHARED_EXPERTS_INPUTS, 
                                                      permute_with_ep, unpermute_with_ep)


def is_less_or_equal_rc2_cann_version():
    cann_starts_with = ('8.0.RC1', '8.0.RC2')
    cann_all = ('not known', '8.0.T1', '8.0.T2', '8.0.T3', '8.0.T37', '8.0.T5', '8.0.T6', '8.0.T7',
                '8.0.T8', '8.0.T10', '8.0.T13', '8.0.T16', '8.0.T50', '8.0.T51', '8.0.T52')
    cann_version = get_cann_version()
    return cann_version in cann_all or cann_version.startswith(cann_starts_with)


cann_version_check = is_less_or_equal_rc2_cann_version()


def allgather_token_permutation(self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind):
    args = get_args()
    self.hidden_shape = hidden_states.shape
    # [S/TP, B, H] -> [S*B/TP, H]
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

    # Permute the tokens across the expert parallel devices.
    if (self.config.tensor_model_parallel_size > 1) or (
            self.config.expert_model_parallel_size > 1
    ):
        # [S*B/TP, H] -> [S*B, H]
        with torch.no_grad():
            global_indices, gi_handle = max_ind if isinstance(max_ind,
                                                              tuple) else gather_from_sequence_parallel_region_to_moe_async(
                max_ind)
        global_probs, gp_handle = gather_from_sequence_parallel_region_to_moe_async(max_prob)
        global_hidden_states, ghs_handle = gather_from_sequence_parallel_region_to_moe_async(hidden_states)

        with torch.no_grad():
            gi_handle.wait()
            global_local_mask = (global_indices >= self.local_expert_indices[0]) & \
                                (global_indices <= self.local_expert_indices[-1])
            local_indices = global_indices.masked_select(global_local_mask)
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
        self.all_tokens_per_expert = all_tokens_per_expert.to(torch.long)
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
            # use 0.7.0 implement for better performance
            tokens_per_expert = torch.histc(
                local_indices,
                bins=self.num_local_experts,
                min=self.local_expert_indices[0],
                max=self.local_expert_indices[-1],
            )
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
            unpermuted_local_hidden.index_put_((self.indices,), hidden_states[:self.indices.shape[0], :],
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
            unpermuted_global_hidden = torch.zeros(global_hidden_shape, dtype=torch.float,
                                                   device=torch.cuda.current_device())
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

            output_bias_total = (
                tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                    unpermuted_global_bias
                )
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
                                                                  unpermuted_local_hidden[
                                                                  :self.global_local_map.shape[0], :],
                                                                  accumulate=True)
            else:
                output_total = unpermuted_global_hidden.scatter_add(
                    0, self.global_local_map, unpermuted_local_hidden
                )
            if self.add_bias:
                unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                if cann_version_check:
                    output_bias_total = unpermuted_global_bias.index_put((self.global_local_map,),
                                                                         unpermuted_local_bias[
                                                                         :self.global_local_map.shape[0], :],
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
    # use 0.7.0 implement for better performance
    num_local_tokens_per_expert = torch.histc(
        indices, bins=self.num_experts, min=0, max=self.num_experts
    )
    # num_local_tokens_per_expert: [num_experts]

    ep_size = self.config.expert_model_parallel_size
    if self.drop_and_pad:
        # probs: [num_experts, capacity]
        self.capacity = self.probs.size(1)
        num_tokens_per_local_expert = torch.full(
            (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long,
            device=torch.cuda.current_device()
        )
        return num_tokens_per_local_expert
    elif self.config.moe_expert_capacity_factor is not None:
        # Token drop but no pad. A synchronization is needed before the first
        # permutation to get the `num_out_tokens` CPU value.
        self.num_out_tokens = num_local_tokens_per_expert.sum().to(
            torch.device("cpu"), non_blocking=True
        )
        self.cuda_sync_point = "before_permutation_1"
    elif ep_size > 1:
        # Token dropless and enable ep. A synchronization is needed before expert parallel
        # AlltoAll communication to get the `input_splits` and `output_splits` CPU values.
        self.cuda_sync_point = "before_ep_alltoall"
    else:
        # Token dropless and no ep. A synchronization is needed before the token_permutation()
        # function returns to get the `tokens_per_expert` CPU value.
        self.cuda_sync_point = "before_finish"

    if ep_size > 1:
        # ===================================================
        # Calculate input_splits, output_splits for alltoall-v.
        # ===================================================
        self.input_splits = (
            num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(
            num_local_tokens_per_expert
        ).reshape(ep_size, self.num_experts)
        self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                                                  :, self.local_expert_indices[0]: self.local_expert_indices[-1] + 1
                                                  ]
        self.output_splits = (
            self.num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu")).numpy()
        )
        num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0)
        # ===================================================
        # num_global_tokens_per_expert: [ep_size, num_experts]
        # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
        # num_tokens_per_local_expert: [num_local_experts]
        # ===================================================
    else:
        self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
            -1, self.num_experts
        )
        num_tokens_per_local_expert = num_local_tokens_per_expert

    if self.num_local_experts > 1:
        if not hasattr(self, 'comm_stream'):
            self.comm_stream = torch.cuda.Stream()
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.comm_stream):
            # No further synchronization is needed because torch.repeat_interleave() calls stream
            # synchronization internally when the `output_size` parameter is not provided.
            self.cuda_sync_point = "no_sync"
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
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
    if self.cuda_sync_point == "before_permutation_1":
        torch.cuda.current_stream().synchronize()
    permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
        hidden_states,
        indices,
        num_out_tokens=self.num_out_tokens,
        padded_mode=self.drop_and_pad,
    )

    if get_args().moe_bmm_mc2:
        return permutated_local_input_tokens, tokens_per_expert

    # Perform expert parallel AlltoAll communication
    if self.cuda_sync_point == "before_ep_alltoall":
        torch.cuda.current_stream().synchronize()
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
    if self.cuda_sync_point == "before_finish":
        torch.cuda.current_stream().synchronize()

    return global_input_tokens, tokens_per_expert


def alltoall_token_unpermutation_with_bmm(
    self, hidden_states: torch.Tensor, bias: torch.Tensor = None,
):
    # if use op bmm_reducescatter_alltoall to skip reducescatter and alltoall
    output = unpermute(
        hidden_states,
        self.reversed_local_input_permutation_mapping,
        probs=self.probs,
        padded_mode=self.drop_and_pad,
        restore_shape=self.hiddden_shape_before_permute,
    )

    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        output = tensor_parallel.all_to_all_hp2sp(output)

    output = output.view(self.hidden_shape)
    return output, None


def alltoall_token_permutation_with_bmm(
    self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor,
):
    # if use op alltoall_allgather_bmm to skip alltoall and allgather
    self.hidden_states = hidden_states.shape
    self.probs = probs
    assert probs.dim() == 2, "Experted 2D tensor for probs"
    assert indices.dim() == 2, "Experted 2D tensor for indices"
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
    tokens_per_expert = self.preprocess(indices)

    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)

    self.hidden_shape_before_permute = hidden_states.shape
    permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
        hidden_states,
        indices,
        num_out_tokens=self.num_out_tokens,
        padded_mode=self.drop_and_pad,
    )
    return permutated_local_input_tokens, tokens_per_expert


def preprocess_tp_extend_ep(self, indices: torch.Tensor, *args) -> torch.Tensor:
    moe_hierarchical_alltoallv = get_args().moe_hierarchical_alltoallv
    num_local_tokens_per_expert = torch.histc(
        indices, bins=self.num_experts, min=0, max=self.num_experts
    )
    # num_local_tokens_per_expert: [num_experts]

    ep_size = self.config.expert_model_parallel_size
    if self.drop_and_pad:
        # probs: [num_experts, capacity]
        self.capacity = self.probs.size(1)
        num_tokens_per_local_expert = torch.full(
            (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long,
            device=torch.cuda.current_device()
        )
        return num_tokens_per_local_expert
    elif self.config.moe_expert_capacity_factor is not None:
        self.num_out_tokens = num_local_tokens_per_expert.sum().cpu()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_extended_ep_size = ep_size * tp_size
    if tp_extended_ep_size > 1:
        # ===================================================
        # Calculate input_splits, output_splits for alltoall-v.
        # ===================================================
        if moe_hierarchical_alltoallv:
            tp_group = parallel_state.get_tensor_model_parallel_group()
            self.input_splits_tp_ep = (
                num_local_tokens_per_expert.reshape(tp_extended_ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"))
                .numpy()
            )
            expert_parallel_rank = mpu.get_expert_model_parallel_rank()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            offset = expert_parallel_rank * tp_size
            self.input_splits = [self.input_splits_tp_ep[i + offset] for i in range(tp_size)]
            self.input_splits_tp_ep = self.input_splits_tp_ep.reshape(ep_size, tp_size).sum(axis=1)
            num_global_tokens_per_expert = \
                all_gather(num_local_tokens_per_expert, group=tp_group).reshape(tp_size, self.num_experts)
            # shared_experts allgather with tp
            if get_args().n_shared_experts and parallel_state.get_tensor_model_parallel_world_size() > 1:
                _, shared_experts_input, shared_experts_allgather_handle = async_all_gather(
                    args[0], parallel_state.get_tensor_model_parallel_group(), is_use_get_global_memory_buffer=True
                )
                AG_SHARED_EXPERTS_INPUTS.append((shared_experts_input, shared_experts_allgather_handle))
        else:
            self.input_splits_tp_ep = None
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
        num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0)
        # ===================================================
        # num_global_tokens_per_expert: [ep_size, num_experts]
        # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
        # num_tokens_per_local_expert: [num_local_experts]
        # ===================================================
    else:
        self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
            -1, self.num_experts
        )
        num_tokens_per_local_expert = num_local_tokens_per_expert

    if self.num_local_experts > 1:
        if not hasattr(self, 'comm_stream'):
            self.comm_stream = torch.cuda.Stream()
        self.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.comm_stream):
            if moe_hierarchical_alltoallv:
                expert_ids_per_ep_rank = torch.tensor(
                    [i % self.num_local_experts for i in range(self.config.num_moe_experts // ep_size)],
                    dtype=torch.int32,
                    device=torch.cuda.current_device(),
                )
            else:
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
        parallel_state.get_tensor_and_expert_parallel_group(),
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
        if not self.drop_and_pad:
            hidden_states = unpermute(
                hidden_states, self.reversed_global_input_permutation_mapping,
            )
        else:
            hidden_states = hidden_states.reshape(
                self.num_local_experts, self.ep_size, self.capacity, -1
            )
            hidden_states = (
                hidden_states.transpose(0, 1)
                .reshape(self.ep_size * self.num_local_experts * self.capacity, -1)
                .contiguous()
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
        probs=self.probs,
        padded_mode=self.drop_and_pad,
        restore_shape=self.hiddden_shape_before_permute,
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

    if (self.config.tensor_model_parallel_size > 1) or (
            self.config.expert_model_parallel_size > 1
    ):
        with (torch.no_grad()):
            gi_handle.wait()
            global_local_mask = (global_indices >= self.local_expert_indices[0]) & \
                                (global_indices <= self.local_expert_indices[-1])

            # masked_select -> reshape
            local_indices = global_indices.masked_select(global_local_mask)
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
        self.all_tokens_per_expert = all_tokens_per_expert.to(torch.long)
        tokens_per_expert = self.all_tokens_per_expert[self.local_expert_indices[0]: self.local_expert_indices[-1] + 1]
        self.global_local_map = global_local_mask.nonzero()[:, 0]

        if self.router_topk > 1:  # k > 1
            gp_handle.wait()
            # masked_select -> reshape
            self.local_probs = global_probs.masked_select(global_local_mask)

        ghs_handle.wait()
        if cann_version_check:
            local_hidden_states = global_hidden_states[self.global_local_map, :]
        else:
            self.global_local_map = self.global_local_map.view(-1, 1).expand(-1, self.hidden_shape[-1])
            local_hidden_states = moe_gather.apply(global_hidden_states, self.global_local_map)
    if self.num_local_experts > 1:
        if cann_version_check:
            permuted_local_hidden_states = local_hidden_states[self.indices, :]
        else:
            self.indices = self.indices.view(-1, 1).expand(-1, self.hidden_shape[-1])
            permuted_local_hidden_states = moe_gather.apply(local_hidden_states, self.indices)
    else:
        permuted_local_hidden_states = local_hidden_states
    return (
        permuted_local_hidden_states,
        tokens_per_expert,
        self.global_local_map,
        self.indices
    )


def allgather_token_unpermutation_new(self, hidden_states: torch.Tensor, bias: torch.Tensor = None):
    # Stage1: unpermute the tokens and bias locally respectively.w
    scores = self.local_probs.to(dtype=hidden_states.dtype)
    if self.num_local_experts > 1:
        if cann_version_check:
            unpermuted_local_hidden = torch.zeros_like(hidden_states)
            unpermuted_local_hidden.index_put_((self.indices,), hidden_states[:self.indices.shape[0], :],
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
        ), "global_local_map is necessary for 'AllGather'."
        ep_group_size = parallel_state.get_tensor_and_expert_parallel_world_size()
        # hidden_shape: [SeqLen/TP, MBS, HiddenSize], global_num_tokens = SeqLen/TP*MBS*(TP*EP)
        global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * ep_group_size
        global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]

        if cann_version_check:
            unpermuted_global_hidden = torch.zeros(global_hidden_shape, dtype=torch.float,
                                                   device=torch.cuda.current_device())
            unpermuted_global_hidden = NewIndePut.apply(unpermuted_global_hidden, (self.global_local_map,),
                                                        unpermuted_local_hidden[:self.global_local_map.shape[0], :])
        else:
            unpermuted_global_hidden = torch.zeros(
                global_hidden_shape, dtype=hidden_states.dtype, device=torch.cuda.current_device()
            )
            # Reshape global_local_map to be compatible with Tensor.scatter
            assert self.global_local_map.shape == unpermuted_local_hidden.shape
            unpermuted_global_hidden = unpermuted_global_hidden.scatter_add(
                0, self.global_local_map, unpermuted_local_hidden)

        output_total = unpermuted_global_hidden
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
                device=torch.cuda.current_device()
            )
            if cann_version_check:
                output_total = unpermuted_global_hidden.index_put((self.global_local_map,),
                                                                  unpermuted_local_hidden[
                                                                  :self.global_local_map.shape[0], :],
                                                                  accumulate=True)
            else:
                output_total = unpermuted_global_hidden.scatter_add(
                    0, self.global_local_map, unpermuted_local_hidden
                )

            if self.add_bias:
                unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                if cann_version_check:
                    output_bias_total = unpermuted_global_bias.index_put((self.global_local_map,),
                                                                         unpermuted_local_bias[
                                                                         :self.global_local_map.shape[0], :],
                                                                         accumulate=True)
                else:
                    output_bias_total = unpermuted_global_bias.scatter_add(
                        0, self.global_local_map, unpermuted_local_bias
                    )

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
        self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor, shared_experts, save_tensors, shared_expert_gate, moe_ctx=None
):
    moe_hierarchical_alltoallv = get_args().moe_hierarchical_alltoallv
    self.hidden_shape = hidden_states.shape
    self.probs = probs
    assert probs.dim() == 2, "Expected 2D tensor for probs"
    assert indices.dim() == 2, "Expected 2D tensor for indices"
    if moe_hierarchical_alltoallv:
        ep_group = parallel_state.get_expert_model_parallel_group()
        _, indices, indices_handle = async_all_gather(indices, group=ep_group)
        indices_handle.wait()
        save_tensors.append(indices)
        _, hidden_states_ep, hidden_states_ep_handle = async_all_gather(hidden_states, group=ep_group)
    else:
        indices_ep, hidden_states_ep, hidden_states_ep_handle = None, None, None
        save_tensors.append(indices_ep)

    def alltoall_token_permutation1(hidden_states, indices, *args):
        if moe_hierarchical_alltoallv:
            _, self.probs, probs_handle = async_all_gather(self.probs, group=ep_group)
            tokens_per_expert = self.preprocess(indices, hidden_states)
            args[1].wait()  # hidden_states_ep_handle
            save_tensors.append(args[0])  # hidden_states_ep
            # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
            hidden_states = args[0].view(-1, self.hidden_shape[-1])
            self.hidden_shape_before_permute = hidden_states.shape
            # Permutation 1: input to AlltoAll input
            if self.cuda_sync_point == "before_permutation_1":
                torch.cuda.current_stream().synchronize()
            probs_handle.wait()
            self.probs = self.probs.detach()
            self.probs.requires_grad = True
            save_tensors.append(self.probs)
            permutated_local_input_tokens, permuted_probs, self.reversed_local_input_permutation_mapping = permute_with_ep(
                hidden_states, indices, probs=self.probs, topk=self.router_topk,
                gb_inputs_splits=self.input_splits_tp_ep,
            )
            self.permuted_probs = permuted_probs
        else:
            tokens_per_expert = self.preprocess(indices)
            save_tensors.append(args[0])
            if get_args().moe_experts_pipeline_degree:
                tokens_per_expert = tokens_per_expert.cpu()

            # Flatten the input tensor
            # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
            hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

            # Perform tensor parallel AlltoAll communication
            # hidden_states: [S*B/TP, H] -> [S*B, H/TP]
            if not get_args().moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
                hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)

            # Permutation 1: input to AlltoAll input
            self.hiddden_shape_before_permute = hidden_states.shape
            if self.cuda_sync_point == "before_permutation_1":
                torch.cuda.current_stream().synchronize()
            scores_ep = None
            save_tensors.append(scores_ep)
            permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
                hidden_states,
                indices,
                num_out_tokens=self.num_out_tokens,
                padded_mode=self.drop_and_pad,
            )
        return tokens_per_expert, permutated_local_input_tokens

    (tokens_per_expert, permutated_local_input_tokens), *_ = forward_func(alltoall_token_permutation1,
                                                                          (hidden_states, indices,
                                                                           hidden_states_ep, hidden_states_ep_handle))

    # permute 1
    save_tensors.append(permutated_local_input_tokens)

    # Perform expert parallel AlltoAll communication
    ep_group = parallel_state.get_expert_model_parallel_group()
    if get_args().moe_tp_extend_ep:
        ep_group = parallel_state.get_tensor_and_expert_parallel_group()

    # Perform expert parallel AlltoAll communication
    if self.cuda_sync_point == "before_ep_alltoall":
        torch.cuda.current_stream().synchronize()
    if moe_hierarchical_alltoallv:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            tp_group,
        )
    else:
        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            ep_group,
        )

    # shared experts
    if shared_experts is not None:
        (share_experts_output, _), *_ = forward_func(shared_experts, (hidden_states, moe_ctx))
        if parallel_state.get_tensor_model_parallel_world_size() > 1 and shared_expert_gate is None:
            share_experts_graph, share_experts_output, rs_shared_experts_handle = async_reduce_scatter(share_experts_output, parallel_state.get_tensor_model_parallel_group(),
                                                                                                       event=permute1_ep_all_to_all_handle, stream=torch.npu.default_stream())
            share_experts_output = (share_experts_graph, share_experts_output, rs_shared_experts_handle)
        if shared_expert_gate is not None:
            with torch.enable_grad():
                # tp not support shared expert gate for now
                if parallel_state.get_tensor_model_parallel_world_size() > 1:
                    share_experts_output = reduce_scatter_to_sequence_parallel_region(share_experts_output)
                share_experts_output = torch.nn.functional.sigmoid(shared_expert_gate(hidden_states)) * share_experts_output
    else:
        share_experts_output = None

    if permute1_ep_all_to_all_handle is not None:
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

    def alltoall_token_permutation2(global_input_tokens):
        # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
        if self.num_local_experts > 1:
            if not self.drop_and_pad:
                if self.comm_stream is not None:
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
        # Perform tensor parallel AllGather on the hidden dimension to obtain the input tokens.
        # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
        need_tp_comm = (not get_args().moe_tp_extend_ep and
                        parallel_state.get_tensor_model_parallel_world_size() > 1 and
                        self.config.moe_grouped_gemm) and get_args().moe_experts_pipeline_degree == 0
        if need_tp_comm:
            global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
                global_input_tokens
            )
        if self.cuda_sync_point == "before_finish":
            torch.cuda.current_stream().synchronize()

        return global_input_tokens

    # token 重排2 input
    (global_input_tokens), global_input_tokens_detach = forward_func(alltoall_token_permutation2,
                                                                     global_input_tokens)
    save_tensors.append(global_input_tokens_detach)
    save_tensors.append(global_input_tokens)
    global_input_tokens_detach.untyped_storage().resize_(0)

    return share_experts_output, global_input_tokens, tokens_per_expert


def alltoall_token_unpermutation_new(
        self, hidden_states, bias, save_tensors
):
    moe_hierarchical_alltoallv = get_args().moe_hierarchical_alltoallv

    def alltoall_token_unpermutation1(hidden_states):
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        # Perform tensor parallel Reduce-Scatter
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        if not get_args().moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1 and get_args().moe_experts_pipeline_degree == 0:
            hidden_states = tensor_parallel.reduce_scatter_last_dim_to_tensor_parallel_region(hidden_states)

        # Unpermutation 2: expert output to AlltoAll input
        if self.num_local_experts > 1:
            if not self.drop_and_pad:
                hidden_states = unpermute(
                    hidden_states, self.reversed_global_input_permutation_mapping,
                )
            else:
                hidden_states = hidden_states.reshape(
                    self.num_local_experts, self.ep_size, self.capacity, -1
                )
                hidden_states = (
                    hidden_states.transpose(0, 1)
                    .reshape(self.ep_size * self.num_local_experts * self.capacity, -1)
                    .contiguous()
                )
        return hidden_states
    if get_args().moe_experts_pipeline_degree:
        with torch.enable_grad():
            hidden_states = alltoall_token_unpermutation1(hidden_states)
        save_tensors.append(hidden_states)
    else:
        hidden_states, unpermute1_input_detach = forward_func(alltoall_token_unpermutation1, hidden_states)
        save_tensors.append(unpermute1_input_detach)
        save_tensors.append(hidden_states)
        unpermute1_input_detach.untyped_storage().resize_(0)

    ep_group = parallel_state.get_expert_model_parallel_group()
    if get_args().moe_tp_extend_ep:
        ep_group = parallel_state.get_tensor_and_expert_parallel_group()
    # Perform expert parallel AlltoAll communication
    # hidden_states: [SEQL, H] -> [SEQL, H/TP]
    if moe_hierarchical_alltoallv:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            self.input_splits,
            self.output_splits,
            tp_group
        )
    else:
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            self.input_splits,
            self.output_splits,
            ep_group
        )
    if handle is not None:
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

    def alltoall_token_unpermutation2(permutated_local_input_tokens):
        # Unpermutation 1: AlltoAll output to output
        if get_args().moe_zero_memory != "disable":
            output = UnpermuteWithoutActivation.apply(
                permutated_local_input_tokens,
                self.reversed_local_input_permutation_mapping,
                self.probs
            )
        else:
            if moe_hierarchical_alltoallv:
                unpermute_with_ep_input_tensors_list = [permutated_local_input_tokens,
                                                        self.reversed_local_input_permutation_mapping,
                                                        self.permuted_probs]
                output = unpermute_with_ep(
                    unpermute_with_ep_input_tensors_list,
                    restore_shape=self.hidden_shape_before_permute,
                    probs=self.probs,
                    topk=self.router_topk
                )
            else:
                output = unpermute(
                    permutated_local_input_tokens,
                    self.reversed_local_input_permutation_mapping,
                    probs=self.probs,
                    padded_mode=self.drop_and_pad,
                    restore_shape=self.hiddden_shape_before_permute,
                )
        if moe_hierarchical_alltoallv:
            return output
        # Perform tensor parallel AlltoAll communication
        # output: [S*B, H/TP] -> [S*B/TP, H]
        if not get_args().moe_tp_extend_ep and parallel_state.get_tensor_model_parallel_world_size() > 1:
            output = tensor_parallel.all_to_all_hp2sp(output)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output

    output, unpermute2_input_detach = forward_func(alltoall_token_unpermutation2, permutated_local_input_tokens)
    save_tensors.append(unpermute2_input_detach)
    should_resize = not self.drop_and_pad and not moe_hierarchical_alltoallv and \
                    not get_args().use_fused_moe_token_permute_and_unpermute or get_args().moe_zero_memory != "disable"
    if should_resize:
        unpermute2_input_detach.untyped_storage().resize_(0)
    save_tensors.append(output)

    if moe_hierarchical_alltoallv:
        ep_group = parallel_state.get_expert_model_parallel_group()
        _, output, output_handle = async_reduce_scatter(output, group=ep_group)
        output_handle.wait()
        output = output.view(self.hidden_shape)
    return output, None


def allgather_token_permutation_npu(self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind: torch.Tensor):
    self.hidden_shape = hidden_states.shape
    # [S/TP, B, H] -> [S*B/TP, H]
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

    # Permute the tokens across the expert parallel devices.
    if (self.config.tensor_model_parallel_size > 1) or (
            self.config.expert_model_parallel_size > 1
    ):
        with torch.no_grad():
            global_indices = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                max_ind
            )
            # Create a mask of mapping between global and local tokens where each
            # element is True if it's between the local_expert_indices
            global_local_mask = (global_indices >= self.local_expert_indices[0]) & (
                    global_indices <= self.local_expert_indices[-1]
            )
            local_indices = global_indices.masked_select(global_local_mask)

        if self.router_topk > 1:  # k > 1
            global_probs = tensor_parallel.gather_from_sequence_parallel_region_to_moe(max_prob)
            self.local_probs = global_probs.masked_select(global_local_mask)
        else:
            self.local_probs = max_prob

        # [S*B/TP, H] -> [S*B, H]
        global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
            hidden_states, use_global_buffer=True
        )
        # Reshape global_local_mask to be compatible with Tensor.gather
        global_local_map = global_local_mask.nonzero()[:, 0]
        self.global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
        local_hidden_states = moe_gather.apply(global_hidden_states, self.global_local_map)
    else:
        if self.router_topk > 1:
            global_local_mask = torch.ones_like(max_ind).bool()
            local_indices = max_ind.masked_select(global_local_mask)
            self.local_probs = max_prob.masked_select(global_local_mask)
            global_local_map = global_local_mask.nonzero()[:, 0]
            self.global_local_map = global_local_map.view(-1, 1).expand(
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
        # use 0.7.0 implement for better performance
        tokens_per_expert = torch.histc(
            local_indices,
            bins=self.num_local_experts,
            min=self.local_expert_indices[0],
            max=self.local_expert_indices[-1],
        )
        tokens_per_expert = tokens_per_expert.to(torch.long)

    # Stage2: permute the tokens locally so that they are grouped by their expert assignment
    # Reshape indices to be compatible with Tensor.gather
    self.indices = self.indices.view(-1, 1).expand(-1, hidden_states.shape[-1])
    if self.num_local_experts > 1:
        permuted_local_hidden_states = moe_gather.apply(local_hidden_states, self.indices)
    else:
        permuted_local_hidden_states = local_hidden_states
    return (
        permuted_local_hidden_states,
        tokens_per_expert,
    )


def alltoall_preprocess_npu(self, indices: torch.Tensor):
    # use 0.7.0 implement for better performance
    num_local_tokens_per_expert = torch.histc(
        indices, bins=self.num_experts, min=0, max=self.num_experts
    )
    # num_local_tokens_per_expert: [num_experts]

    ep_size = self.config.expert_model_parallel_size
    if self.drop_and_pad:
        # probs: [num_experts, capacity]
        self.capacity = self.probs.size(1)
        num_tokens_per_local_expert = torch.full(
            (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long,
            device=torch.cuda.current_device()
        )
        return num_tokens_per_local_expert
    elif self.config.moe_expert_capacity_factor is not None:
        # Token drop but no pad.
        self.num_out_tokens = num_local_tokens_per_expert.sum().to(
            torch.device("cpu"), non_blocking=True
        )
        self.cuda_sync_point = "before_permutation_1"
    elif ep_size > 1:
        # Token dropless and enable ep.
        self.cuda_sync_point = "before_ep_alltoall"
    else:
        # Token dropless and no ep.
        self.cuda_sync_point = "before_finish"

    if ep_size > 1:
        # ===================================================
        # Calculate input_splits, output_splits for alltoall-v.
        # ===================================================
        self.input_splits = (
            num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )
        num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(
            num_local_tokens_per_expert
        ).reshape(ep_size, self.num_experts)
        self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                                                  :, self.local_expert_indices[0]: self.local_expert_indices[-1] + 1
                                                  ]
        self.output_splits = (
            self.num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu")).numpy()
        )
        num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0)
        # ===================================================
        # num_global_tokens_per_expert: [ep_size, num_experts]
        # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
        # num_tokens_per_local_expert: [num_local_experts]
        # ===================================================
    else:
        self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
            -1, self.num_experts
        )
        num_tokens_per_local_expert = num_local_tokens_per_expert

    if self.num_local_experts > 1:
        # No further synchronization is needed because torch.repeat_interleave() calls stream
        # synchronization internally when the `output_size` parameter is not provided.
        self.cuda_sync_point = "no_sync"
        self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
            self.expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
        )

    return num_tokens_per_local_expert