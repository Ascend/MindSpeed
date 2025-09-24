# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import numpy as np
import mindspore
from torch_npu.utils.collect_env import get_cann_version
from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel, mpu
from megatron.core.transformer.moe.moe_utils import permute, unpermute, sort_chunks_by_idxs, get_capacity
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.tensor_parallel.mappings import reduce_scatter_to_sequence_parallel_region
from mindspeed.core.transformer.moe.router import gather_from_sequence_parallel_region_to_moe_async
from mindspeed.core.transformer.moe.comm_utils import (async_reduce_scatter,
                                                       async_all_gather)
from mindspeed.mindspore.core.transformer.moe.moe_layer_overlap_all2all import forward_func
from mindspeed.core.transformer.moe.unpermute_without_activation import UnpermuteWithoutActivation
from mindspeed.core.transformer.moe.moe_utils import AG_SHARED_EXPERTS_INPUTS
from mindspeed.mindspore.core.transformer.moe.comm_utils import async_all_to_all


def is_less_or_equal_rc2_cann_version():
    cann_starts_with = ('8.0.RC1', '8.0.RC2')
    cann_all = ('not known', '8.0.T1', '8.0.T2', '8.0.T3', '8.0.T37', '8.0.T5', '8.0.T6', '8.0.T7',
                '8.0.T8', '8.0.T10', '8.0.T13', '8.0.T16', '8.0.T50', '8.0.T51', '8.0.T52')
    cann_version = get_cann_version()
    return cann_version in cann_all or cann_version.startswith(cann_starts_with)


cann_version_check = is_less_or_equal_rc2_cann_version()


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
        self.num_out_tokens = num_local_tokens_per_expert.sum()
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
            .numpy()
        )
        num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(
            num_local_tokens_per_expert
        ).reshape(ep_size, self.num_experts)
        self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                                                  :, self.local_expert_indices[0]: self.local_expert_indices[-1] + 1
                                                  ]
        self.output_splits = (
            self.num_global_tokens_per_local_expert.sum(axis=-1).numpy()
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
            self.comm_stream = mindspore.runtime.Stream()
        self.comm_stream.wait_stream(mindspore.runtime.current_stream())
        with mindspore.runtime.StreamCtx(self.comm_stream):
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
        mindspore.runtime.current_stream().synchronize()
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
        mindspore.runtime.current_stream().synchronize()
    global_input_tokens = tensor_parallel.all_to_all(
        parallel_state.get_expert_model_parallel_group(),
        permutated_local_input_tokens,
        self.output_splits,
        self.input_splits,
    )

    # Permutation 2: AlltoAll output to expert input if num_local_experts > 1
    if self.num_local_experts > 1:
        if not self.drop_and_pad:
            mindspore.runtime.current_stream().wait_stream(self.comm_stream)
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
        mindspore.runtime.current_stream().synchronize()

    return global_input_tokens, tokens_per_expert


def moealltoallseqtokendispatcher_init(
    self, num_local_experts: int, local_expert_indices: list[int], config: TransformerConfig
) -> None:
    """
    Initialize the AlltoAll token dispatcher.

    Args:
        num_local_experts (int): Number of local experts on the current device.
        local_expert_indices (list[int]): Indices of local experts on the current device.
        config (TransformerConfig): Configuration for the transformer model.
    """
    super(MoEAlltoAllSEQTokenDispatcher, self).__init__(config=config)
    self.hidden_shape = None
    self.num_input_tokens = None
    self.num_local_experts = num_local_experts
    self.num_experts = config.num_moe_experts
    assert self.num_local_experts > 0, "Expected at least one expert"
    self.local_expert_indices = local_expert_indices
    assert (
        len(self.local_expert_indices) == self.num_local_experts
    ), "Invalid local expert indices"
    for i in range(len(self.local_expert_indices) - 1):
        assert (
            self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
        ), "local_expert_indices must be continous"
    self.ep_size = config.expert_model_parallel_size
    self.tp_size = config.tensor_model_parallel_size
    self.input_splits = None
    self.output_splits = None
    # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
    # to each local expert by all ranks.
    self.num_global_tokens_per_local_expert_cpu = None
    input_chunk_idxs = np.arange(self.num_experts)
    # [num_local_experts, ep_size]. Sort the input chunks by local experts.
    self.sort_input_by_local_experts = input_chunk_idxs.reshape(
        -1, self.num_local_experts
    ).T.ravel()
    # [ep_size, num_local_experts]. Restore the output chunks by local experts.
    self.restore_output_by_local_experts = input_chunk_idxs.reshape(
        self.num_local_experts, -1
    ).T.ravel()

    # Token drop and padding.
    # We need to keep track of the token num if we drop tokens without padding them.
    self.num_out_tokens = None
    # Drop and pad the input to capacity.
    self.drop_and_pad = self.config.moe_pad_expert_input_to_capacity
    if self.drop_and_pad:
        assert self.config.moe_expert_capacity_factor is not None
    self.capacity = None

    # A cuda stream synchronization is needed in self.token_permutation()
    # in some cases, because there are several non-blocking DtoH data
    # transfers called in self.preprocess(). The synchronization happens
    # at different points based on MoE settings as late as possible.
    # Valid sync points are "before_permutation_1", "before_ep_alltoall",
    # "before_finish", and "no_sync".
    self.cuda_sync_point = "no_sync"
