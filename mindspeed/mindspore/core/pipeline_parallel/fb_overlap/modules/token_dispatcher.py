# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import acl
import mindspore
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.moe.moe_utils import permute, unpermute
from megatron.core.utils import make_viewless_tensor
from megatron.training import get_args
from mindspeed.core.pipeline_parallel.fb_overlap.modules.utils import TensorSwapManager

from mindspeed.mindspore.op_builder import AlgorithmOpBuilder


reuse_data_ptr = mindspore.utils._reuse_data_ptr


class StreamConfig:
    stream = None
overlap_stream = StreamConfig


def preprocess(self, indices: torch.Tensor) -> torch.Tensor:
    # use 0.7.0 implement for better performance
    num_local_tokens_per_expert = torch.histc(
        indices, bins=self.num_experts, min=0, max=self.num_experts
    )

    ep_size = self.config.expert_model_parallel_size
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_extended_ep_size = ep_size * tp_size
    if self.drop_and_pad:
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
    elif tp_extended_ep_size > 1:
        # Token dropless and enable ep. A synchronization is needed before expert parallel
        # AlltoAll communication to get the `input_splits` and `output_splits` CPU values.
        self.cuda_sync_point = "before_ep_alltoall"
    else:
        # Token dropless and no ep. A synchronization is needed before the token_permutation()
        # function returns to get the `tokens_per_expert` CPU value.
        self.cuda_sync_point = "before_finish"

    if tp_extended_ep_size > 1:
        # ===================================================
        # Calculate input_splits, output_splits for alltoall-v.
        # ===================================================
        self.input_splits = (
            num_local_tokens_per_expert.reshape(tp_extended_ep_size, self.num_local_experts)
            .sum(axis=1)
            .numpy()
        )
        num_global_tokens_per_expert = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
            num_local_tokens_per_expert
        ).reshape(tp_extended_ep_size, self.num_experts)
        self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ]
        self.output_splits = (
            self.num_global_tokens_per_local_expert
            .sum(axis=-1)
            .numpy()
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

    return num_tokens_per_local_expert


class PackProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permutated_local_input_tokens, probs):
        ctx.prob_dtype = probs.dtype
        ctx.token_dtype = permutated_local_input_tokens.dtype
        if ctx.token_dtype == torch.bfloat16 and ctx.prob_dtype == torch.bfloat16:
            viewd_probs = probs
        else:
            probs_shape_new = probs.shape[:-1] + (probs.shape[-1] * 2,)
            viewd_probs = torch.empty(probs_shape_new, dtype=ctx.token_dtype)
            reuse_data_ptr(viewd_probs, probs, 0)

        ctx.prob_split_sizes = [permutated_local_input_tokens.shape[-1], viewd_probs.shape[-1]]

        packed_tokens = torch.cat((permutated_local_input_tokens, viewd_probs), dim=-1)
        permutated_local_input_tokens.untyped_storage().resize_(0)
        viewd_probs.untyped_storage().resize_(0)

        return packed_tokens


    @staticmethod
    def backward(ctx, grad_output):
        grad_tokens, grad_probs = grad_output.split(ctx.prob_split_sizes, dim=-1)
        if ctx.token_dtype == torch.bfloat16 and ctx.prob_dtype == torch.bfloat16:
            viewd_grad_probs = grad_probs
        else:
            grad_probs_shape = grad_probs.shape[:-1] + (grad_probs.shape[-1] // 2,)
            viewd_grad_probs = torch.empty(grad_probs_shape, dtype=ctx.prob_dtype)
            reuse_data_ptr(viewd_grad_probs, grad_probs, 0)

        return grad_tokens.contiguous(), viewd_grad_probs.contiguous()


class UnpackProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed_tokens, prob_split_sizes, orig_prob_dtype):
        ctx.token_dtype = packed_tokens.dtype
        ctx.orig_prob_dtype = orig_prob_dtype
        tokens, probs = packed_tokens.split(prob_split_sizes, dim=-1)
        if ctx.token_dtype == torch.bfloat16 and orig_prob_dtype == torch.bfloat16:
            viewd_probs = probs
        else:
            probs_shape = probs.shape[:-1] + (probs.shape[-1] // 2,)
            viewd_probs = torch.empty(probs_shape, dtype=orig_prob_dtype)
            reuse_data_ptr(viewd_probs, probs, 0)
            viewd_probs = viewd_probs.contiguous()

        return tokens.contiguous(), viewd_probs

    @staticmethod
    def backward(ctx, grad_tokens, grad_probs):
        if ctx.token_dtype == torch.bfloat16 and ctx.orig_prob_dtype == torch.bfloat16:
            viewd_grad_probs = grad_probs
        else:
            grad_probs_shape = grad_probs.shape[:-1] + (grad_probs.shape[-1] * 2,)
            viewd_grad_probs = torch.empty(grad_probs_shape, dtype=ctx.token_dtype)
            reuse_data_ptr(viewd_grad_probs, grad_probs, 0)

        grad_input = torch.cat((grad_tokens, viewd_grad_probs), dim=-1)

        return grad_input, None, None


def alltoall_token_perm1(
    self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor,
):
    self.hidden_shape = hidden_states.shape
    args = get_args()
    if args.moe_unperm2_mem_optim:
        self.probs = None
    else:
        self.probs = probs
    assert probs.dim() == 2, "Expected 2D tensor for probs"
    assert indices.dim() == 2, "Expected 2D tensor for indices"
    tokens_per_expert = preprocess(self, indices)

    event = torch.npu.current_stream().record_event()
    global overlap_stream
    if overlap_stream.stream is None:
        overlap_stream.stream = torch.npu.Stream()      
    torch.npu.current_stream().wait_event(event)
    # Flatten the input tensor
    # hidden_states: [S/TP, B, H] -> [S*B/TP, H]
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

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
    permuted_local_probs = None
    if args.moe_unperm2_mem_optim:
        permuted_local_probs, _ = permute(
            probs.view(-1, 1),
            indices.view(-1, 1)
        )
        if '910B' in acl.get_soc_name():
            permutated_local_input_tokens = PackProb.apply(permutated_local_input_tokens, permuted_local_probs)
            self.prob_split_sizes = [hidden_states.shape[-1],
                                    permutated_local_input_tokens.shape[-1] - hidden_states.shape[-1]]
            self.orig_prob_format = probs.dtype
            permuted_local_probs = None

    return permutated_local_input_tokens, permuted_local_probs, tokens_per_expert
