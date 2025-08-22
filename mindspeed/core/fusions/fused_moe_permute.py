# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import wraps
from typing import Optional, Tuple

import torch
import torch_npu
from megatron.core.transformer.moe.moe_utils import maybe_move_tensor_to_cpu
from megatron.core.transformer.moe.moe_utils import permute as megatron_permute
from megatron.core.transformer.moe.moe_utils import unpermute as megatron_unpermute
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs
from mindspeed.te.pytorch.permutation import MoePermuteMaskMap, MoeUnpermuteMaskMap


def convert_tensors_to_fp32_if_needed(
        tensor1: Optional[torch.Tensor],
        tensor2: Optional[torch.Tensor]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.dtype]]:
    if not (isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor)):
        return tensor1, tensor2, None

    dtype1, dtype2 = tensor1.dtype, tensor2.dtype

    if (dtype1 == torch.float32) ^ (dtype2 == torch.float32):
        original_dtype = dtype2 if dtype1 == torch.float32 else dtype1

        tensor1 = tensor1.to(torch.float32)
        tensor2 = tensor2.to(torch.float32)

        return tensor1, tensor2, original_dtype

    return tensor1, tensor2, None


def restore_original_dtype(
        tensor1: Optional[torch.Tensor],
        tensor2: Optional[torch.Tensor],
        original_dtype: Optional[torch.dtype]
) -> Tuple[Optional[torch.Tensor], ...]:
    """Restore tensors to their original dtype if needed."""
    if original_dtype is None:
        return tensor1, tensor2

    return (tensor1.to(original_dtype) if tensor1 is not None else tensor1,
            tensor2.to(original_dtype) if tensor2 is not None else tensor2)


def permute(
        tokens,
        routing_map,
        probs: Optional[torch.Tensor] = None,
        num_out_tokens: Optional[int] = None,
        fused: bool = False,
        drop_and_pad: bool = False,
) -> torch.Tensor:
    if fused:
        tokens, probs, original_dtype = convert_tensors_to_fp32_if_needed(tokens, probs)
        permuted_input, permuted_probs, sorted_indices = (
            MoePermuteMaskMap.apply(tokens, routing_map, probs, num_out_tokens, drop_and_pad))
        permuted_input, permuted_probs = restore_original_dtype(permuted_input, permuted_probs, original_dtype)
        return permuted_input, permuted_probs, sorted_indices
    else:
        return megatron_permute(tokens, routing_map, probs=probs, num_out_tokens=num_out_tokens, fused=fused,
                                drop_and_pad=drop_and_pad)


def unpermute(
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        restore_shape: torch.Size,
        probs: torch.Tensor = None,
        routing_map: torch.Tensor = None,
        fused: bool = False,
        drop_and_pad: bool = False,
) -> torch.Tensor:
    if fused:
        return MoeUnpermuteMaskMap.apply(
            permuted_tokens, sorted_indices, restore_shape, probs, routing_map, drop_and_pad)
    else:
        return megatron_unpermute(permuted_tokens, sorted_indices, restore_shape, probs=probs, routing_map=routing_map,
                                  fused=fused, drop_and_pad=drop_and_pad)


def sort_chunks_by_idxs_wrapper(fn):
    @wraps(fn)
    def wrapper(
            input: torch.Tensor,
            split_sizes: torch.Tensor,
            sorted_idxs: torch.Tensor,
            probs: Optional[torch.Tensor] = None,
            fused: bool = False,
    ) -> torch.Tensor:
        # Currently, fused_sort_chunks_by_index is not supported
        return fn(input, split_sizes, sorted_idxs, probs=probs, fused=False)

    return wrapper


def moe_alltoall_token_dispatcher_init_wrapper(fn):
    @wraps(fn)
    def wrapper(
            self, num_local_experts, local_expert_indices, config
    ) -> None:
        fn(self, num_local_experts, local_expert_indices, config)
        # Since fused_sort_chunks_by_index is not currently supported, set self.permute_idx_device to None
        self.permute_idx_device = None
        input_chunk_idxs = torch.arange(
            self.num_experts * self.tp_size, device=self.permute_idx_device
        )
        # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        # [tp_size * ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

    return wrapper


def maybe_dtoh_and_synchronize(
        self, point: str, tokens_per_expert: torch.Tensor = None
) -> torch.Tensor:
    """
    Move all possible GPU tensors to CPU and make a synchronization at the expected point.
    """
    if not self.drop_and_pad:
        if point == self.cuda_dtoh_point:
            # Move all possible GPU tensors to CPU at self.cuda_dtoh_point.
            on_side_stream = torch.cuda.current_stream() != self.cuda_dtoh_stream
            if on_side_stream:
                self.cuda_dtoh_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.cuda_dtoh_stream):
                tokens_per_expert = maybe_move_tensor_to_cpu(
                    tokens_per_expert, record_stream=on_side_stream
                )
                self.input_splits = maybe_move_tensor_to_cpu(
                    self.input_splits, as_numpy=True, record_stream=on_side_stream
                )
                self.output_splits = maybe_move_tensor_to_cpu(
                    self.output_splits, as_numpy=True, record_stream=on_side_stream
                )
                self.output_splits_tp = maybe_move_tensor_to_cpu(
                    self.output_splits_tp, as_numpy=True, record_stream=on_side_stream
                )
                self.num_out_tokens = maybe_move_tensor_to_cpu(
                    self.num_out_tokens, record_stream=on_side_stream
                )
                # Since fused_sort_chunks_by_index is not currently supported, when self.num_local_experts > 1,
                # move self.num_global_tokens_per_local_expert to cpu
                if self.num_local_experts > 1:
                    self.num_global_tokens_per_local_expert = maybe_move_tensor_to_cpu(
                        self.num_global_tokens_per_local_expert, record_stream=on_side_stream
                    )

        if point == self.cuda_sync_point:
            # Synchronize with the dtoh stream at self.cuda_sync_point.
            self.cuda_dtoh_stream.synchronize()

    return tokens_per_expert


def transformer_config_post_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        # Reset moe_permute_fusion to bypass Megatron check.
        if self.moe_token_dispatcher_type == "alltoall_seq":
            ori_moe_permute_fusion = self.moe_permute_fusion
            self.moe_permute_fusion = False
        fn(self)
        if self.moe_token_dispatcher_type == "alltoall_seq":
            self.moe_permute_fusion = ori_moe_permute_fusion
            del ori_moe_permute_fusion

    return wrapper


def alltoall_seq_token_permutation(
    self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dispatch tokens to local experts using AlltoAll communication.

    Args:
        hidden_states (torch.Tensor): Input token embeddings.
        probs (torch.Tensor): Probs of tokens assigned to experts.
            Shape: [num_tokens, num_experts].
        routing_map (torch.Tensor): Mapping of tokens assigned to experts.
            Shape: [num_tokens, num_experts].

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Permuted token embeddings for local experts.
            - Number of tokens per expert.
            - Permuted probs of each token produced by the router.
    """
    # Preprocess: Get the metadata for communication, permutation and computation operations.
    self.hidden_shape = hidden_states.shape
    self.routing_map = routing_map
    assert probs.dim() == 2, "Expected 2D tensor for probs"
    assert routing_map.dim() == 2, "Expected 2D tensor for routing map"
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
    tokens_per_expert = self.preprocess(routing_map)

    # Perform tensor parallel AlltoAll communication
    # hidden_states: [S*B/TP, H] -> [S*B, H/TP]
    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)

    # Permutation 1: input to AlltoAll input
    self.hidden_shape_before_permute = hidden_states.shape
    if self.cuda_sync_point == "before_permutation_1":
        torch.cuda.current_stream().synchronize()
    (
        permutated_local_input_tokens,
        permuted_probs,
        self.reversed_local_input_permutation_mapping,
    ) = permute(hidden_states, routing_map, probs=probs, num_out_tokens=self.num_out_tokens,
                fused=self.config.moe_permute_fusion)

    # Perform expert parallel AlltoAll communication
    if self.cuda_sync_point == "before_ep_alltoall":
        torch.cuda.current_stream().synchronize()
    global_input_tokens = tensor_parallel.all_to_all(
        parallel_state.get_expert_model_parallel_group(),
        permutated_local_input_tokens,
        self.output_splits,
        self.input_splits,
    )
    global_probs = tensor_parallel.all_to_all(
        parallel_state.get_expert_model_parallel_group(),
        permuted_probs,
        self.output_splits,
        self.input_splits,
    )

    # Permutation 2: Sort tokens by local expert.
    if self.num_local_experts > 1:
        global_input_tokens, global_probs = sort_chunks_by_idxs(
            global_input_tokens,
            self.num_global_tokens_per_local_expert_cpu.ravel(),
            self.sort_input_by_local_experts,
            probs=global_probs,
        )

    # Perform tensor parallel AllGather on the hidden dimension to obtain the input tokens.
    # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
            global_input_tokens
        )
    if self.cuda_sync_point == "before_finish":
        torch.cuda.current_stream().synchronize()

    return global_input_tokens, tokens_per_expert, global_probs


def alltoall_seq_token_unpermutation(
    self, hidden_states: torch.Tensor, bias: torch.Tensor = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

    # Perform tensor parallel Reduce-Scatter
    # hidden_states: [SEQL, H] -> [SEQL, H/TP]
    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        hidden_states = tensor_parallel.reduce_scatter_last_dim_to_tensor_parallel_region(
            hidden_states
        )

    # Unpermutation 2: Unsort tokens by local expert.
    if self.num_local_experts > 1:
        hidden_states, _ = sort_chunks_by_idxs(
            hidden_states,
            self.num_global_tokens_per_local_expert_cpu.T.ravel(),
            self.restore_output_by_local_experts,
        )

    # Perform expert parallel AlltoAll communication
    # hidden_states: [SEQL, H] -> [SEQL, H/TP]
    permutated_local_input_tokens = tensor_parallel.all_to_all(
        parallel_state.get_expert_model_parallel_group(),
        hidden_states,
        self.input_splits,
        self.output_splits,
    )

    # Unpermutation 1: AlltoAll output to output
    output = unpermute(
        permutated_local_input_tokens,
        self.reversed_local_input_permutation_mapping,
        restore_shape=self.hidden_shape_before_permute,
        routing_map=self.routing_map,
        fused=self.config.moe_permute_fusion,
    )

    # Perform tensor parallel AlltoAll communication
    # output: [S*B, H/TP] -> [S*B/TP, H]
    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        output = tensor_parallel.all_to_all_hp2sp(output)

    # Reshape the output tensor
    output = output.view(self.hidden_shape)
    return output, None
