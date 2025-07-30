# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import wraps
from typing import Optional

import torch
import torch_npu
from megatron.core.transformer.moe.moe_utils import maybe_move_tensor_to_cpu
from torch_npu import npu_moe_token_unpermute_with_routing_map


def unpermute_wrapper(fn):
    @wraps(fn)
    def wrapper(
            permuted_tokens: torch.Tensor,
            sorted_indices: torch.Tensor,
            restore_shape: torch.Size,
            probs: torch.Tensor = None,
            routing_map: torch.Tensor = None,
            fused: bool = False,
            drop_and_pad: bool = False,
    ) -> torch.Tensor:
        if fused:
            return npu_moe_token_unpermute_with_routing_map(
                permuted_tokens, sorted_indices, restore_shape, probs=probs, routing_map=routing_map,
                drop_and_pad=drop_and_pad)
        else:
            return fn(permuted_tokens, sorted_indices, restore_shape, probs=probs, routing_map=routing_map, fused=fused,
                      drop_and_pad=drop_and_pad)

    return wrapper


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
