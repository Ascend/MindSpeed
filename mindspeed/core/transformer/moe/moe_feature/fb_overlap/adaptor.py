# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0 OR MIT

from contextlib import nullcontext
from copy import copy
from dataclasses import replace
from logging import getLogger
from functools import partial, wraps
import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.tensor_parallel import (
    all_gather_last_dim_from_tensor_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.utils import make_viewless_tensor
from megatron.core import parallel_state
from mindspeed.args_utils import get_full_args as get_args

from .modules.experts import MindSpeedFbOverlapGmmExperts
from .modules.shared_experts import SharedExpertMLPFbOverlap
from .modules.moe_layer import MindSpeedFbOverlapMoELayer
from .vpp_schedules import forward_backward_pipelining_with_interleaving
from .no_pipelining_schedules import forward_backward_no_pipelining


def _get_te_grouped_mlp_submodules(experts_builder):
    """Find GroupedMLPSubmodules through MindSpeed's nested partial wrappers."""
    from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP

    pending = [experts_builder]
    visited = set()
    found_te_builder = False
    grouped_mlp_submodules = None

    while pending:
        builder = pending.pop()
        builder_id = id(builder)
        if builder_id in visited:
            continue
        visited.add(builder_id)

        if builder is TEGroupedMLP:
            found_te_builder = True
            continue
        if not isinstance(builder, partial):
            continue

        if builder.func is TEGroupedMLP:
            found_te_builder = True
        submodules = (builder.keywords or {}).get('submodules')
        if isinstance(submodules, GroupedMLPSubmodules):
            grouped_mlp_submodules = submodules

        pending.append(builder.func)
        pending.extend(arg for arg in builder.args if isinstance(arg, partial))

    if not found_te_builder or grouped_mlp_submodules is None:
        raise TypeError(
            'MindSpeed --moe-fb-overlap requires a Megatron TEGroupedMLP builder '
            f'with GroupedMLPSubmodules, but got {experts_builder!r}.'
        )
    return grouped_mlp_submodules


def _make_backward_post_hook(self, param: torch.nn.Parameter):
    """
    Creates a backward post-hook to dispatch an all-reduce / reduce-scatter when
    ready (i.e., when all grads in a bucket have been computed in all microbatches
    in a batch).
    """

    def hook(*unused):
        if is_graph_capturing():
            return
        if param in self.param_to_bucket_group:
            if not getattr(param, 'skip_grad_accum', False):
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    assert param.grad is not None, 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None

            if self.ddp_config.overlap_grad_reduce:
                self.param_to_bucket_group[param].register_grad_ready(param, self.force_all_reduce)

        if getattr(param, 'skip_grad_accum', False):
            param.skip_grad_accum = False

    return hook


def fb_overlap_ddp_init_wrapper(fn):
    """Attach DDP hooks to the two TE linears whose wgrad is delayed by FB overlap."""

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)
        if self.ddp_config.delay_wgrad_compute:
            raise RuntimeError(
                'MindSpeed FB overlap owns the TE expert delayed-wgrad schedule '
                'and cannot be combined with global DDP delay_wgrad_compute.'
            )

        marker = '_mindspeed_fb_overlap_delayed_wgrad_hook_registered'
        for experts in self.module.modules():
            if not isinstance(experts, MindSpeedFbOverlapGmmExperts):
                continue

            for module in (experts.linear_fc1, experts.linear_fc2):
                if not (
                    hasattr(module, 'need_backward_dw')
                    and module.need_backward_dw()
                    and hasattr(module, 'register_wgrad_accumulation_and_reduce_hooks')
                ):
                    raise RuntimeError(f'MindSpeed FB overlap requires delayed-wgrad hooks on {type(module).__name__}.')

                for param in module.parameters(recurse=False):
                    if not (
                        param.requires_grad
                        and getattr(param, 'skip_backward_post_hook', False)
                        and getattr(param, marker, None) != id(self)
                    ):
                        continue

                    ddp_post_hook = self._make_backward_post_hook(param)

                    def delayed_wgrad_post_hook(
                        param=param,
                        module=module,
                        ddp_post_hook=ddp_post_hook,
                    ):
                        # With fused accumulation, TE has already accumulated
                        # directly into main_grad and deliberately leaves
                        # param.grad unset. The normal DDP hook must therefore
                        # not try to add a dummy gradient a second time.
                        if param.grad is None and getattr(module, 'fuse_wgrad_accumulation', False):
                            if not hasattr(param, 'main_grad'):
                                raise RuntimeError('TE delayed fused wgrad completed without param.main_grad.')
                            param.grad_added_to_main_grad = True
                            if self.ddp_config.overlap_grad_reduce and param in self.param_to_bucket_group:
                                self.param_to_bucket_group[param].register_grad_ready(param, self.force_all_reduce)
                            return
                        ddp_post_hook()

                    module.register_wgrad_accumulation_and_reduce_hooks(delayed_wgrad_post_hook)
                    setattr(param, marker, id(self))
        return result

    return wrapper


def get_forward_backward_func_vpp_overlap_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        global_args = get_args()
        origin_forward_backward_func = fn(*args, **kwargs)
        # Use moe-fb-overlap customized schedules only for training. Dispatch at
        # call time to keep forward-only eval on Megatron's native schedule.
        if not (torch.is_grad_enabled() and global_args.moe_fb_overlap):
            return origin_forward_backward_func

        pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        overlap_forward_backward_func = (
            forward_backward_pipelining_with_interleaving
            if pipeline_model_parallel_size > 1
            else forward_backward_no_pipelining
        )

        @wraps(origin_forward_backward_func)
        def dispatch_forward_backward_func(*fb_args, **fb_kwargs):
            if fb_kwargs.get('forward_only', False):
                return origin_forward_backward_func(*fb_args, **fb_kwargs)
            return overlap_forward_backward_func(*fb_args, **fb_kwargs)

        return dispatch_forward_backward_func

    return wrapper


def get_moe_module_spec_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        spec = fn(*args, **kwargs)
        args = get_args()
        if args.moe_fb_overlap:
            log = getLogger(__name__)
            log.info("moe_fb_overlap is enabled. Replacing default megatron layer spec for moe...")

            if not isinstance(spec, partial):
                raise TypeError(
                    f'Megatron v0.18 MoE spec must be a functools.partial MlpBuilder, but got {type(spec)!r}.'
                )
            moe_submodules = (spec.keywords or {}).get('submodules')
            if moe_submodules is None:
                raise TypeError("Megatron v0.18 MoE spec must provide partial.keywords['submodules'].")
            from megatron.core.transformer.spec_utils import get_submodules

            grouped_mlp_submodules = _get_te_grouped_mlp_submodules(moe_submodules.experts)

            def fb_overlap_experts_builder(
                num_local_experts,
                config,
                /,
                *,
                pg_collection=None,
                name=None,
            ):
                expert_config = copy(config)
                expert_config.gradient_accumulation_fusion = bool(
                    getattr(config, 'gemm_gradient_accumulation_fusion', False)
                )
                # TE backward computes dgrad immediately and queues wgrad. The
                # existing FB scheduler executes backward_dw() at the old
                # WeightGradStore.pop(experts_only=True) point.
                expert_config.delay_wgrad_compute = True
                # The thin FB-specific forward calls linear_fc1/linear_fc2
                # directly, so fused children must not own backward_dw().
                expert_config.use_transformer_engine_op_fuser = False
                return MindSpeedFbOverlapGmmExperts(
                    num_local_experts,
                    expert_config,
                    submodules=grouped_mlp_submodules,
                    pg_collection=pg_collection,
                    name=name,
                )

            mlp_submodules = get_submodules(moe_submodules.shared_experts)
            shared_builder = partial(SharedExpertMLPFbOverlap, submodules=mlp_submodules)
            new_keywords = dict(spec.keywords or {})
            new_keywords['submodules'] = replace(
                moe_submodules,
                experts=fb_overlap_experts_builder,
                shared_experts=shared_builder,
            )
            new_spec = partial(MindSpeedFbOverlapMoELayer, *spec.args, **new_keywords)
            new_spec.__dict__.update(
                {key: value for key, value in spec.__dict__.items() if key not in {'module', 'submodules'}}
            )
            spec = new_spec

        return spec

    return wrapper


def dualpipev_fb_overlap_mtp_layer_forward_te_without_overlap(
    self,
    decoder_input: Tensor,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Tensor = None,
    context_mask: Tensor = None,
    rotary_pos_emb: Tensor = None,
    rotary_pos_cos: Tensor = None,
    rotary_pos_sin: Tensor = None,
    attention_bias: Tensor = None,
    inference_params=None,
    packed_seq_params=None,
    sequence_len_offset: Tensor = None,
):
    """
    Perform the forward pass through the MTP layer.

    Args:
        hidden_states (Tensor): hidden states tensor of shape [s, b, h] where s is the
            sequence length, b is the batch size, and h is the hidden size.
        decoder_input (Tensor): Input tensor of shape [s, b, h] where s is the
            sequence length, b is the batch size, and h is the hidden size.
            At the (k - 1)-th MTP module, the i-th element of decoder input is
            the embedding of (i + K)-th tocken.
        attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
            self-attention.
        context (Tensor, optional): Context tensor for cross-attention.
        context_mask (Tensor, optional): Mask for cross-attention context
        rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
        attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
            to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
            Used as an alternative to apply attention mask for TE cuDNN attention.
        inference_params (InferenceParams, optional): Parameters for inference-time
            optimizations.
        packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
            processing.

    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
        [s, b, h], and optionally the updated context tensor if cross-attention is used.
    """
    assert context is None, "multi token prediction + cross attention is not yet supported."
    assert packed_seq_params is None, "multi token prediction + sequence packing is not yet supported."

    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

    if self.config.sequence_parallel:
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
    else:
        rng_context = nullcontext()

    if self.config.fp8:
        fp8_context = get_fp8_context(self.config)
    else:
        fp8_context = nullcontext()

    with rng_context, fp8_context:
        decoder_input = self.enorm(decoder_input)
        decoder_input = make_viewless_tensor(inp=decoder_input, requires_grad=True, keep_graph=True)
        hidden_states = self.hnorm(hidden_states)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        # At the (k - 1)-th MTP module, concatenates the i-th tocken's hidden_states
        # and the (i + K)-th tocken's embedding, and combine them with linear projection.
        hidden_states = torch.cat((decoder_input, hidden_states), -1)
        hidden_states, _ = self.eh_proj(hidden_states)
        # For tensor parallel, all gather after linear_fc.
        hidden_states = all_gather_last_dim_from_tensor_parallel_region(hidden_states)
        # For sequence parallel, scatter after linear_fc and before transformer layer.
        if self.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        hidden_states, _ = self.transformer_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

    # Layer norm before shared head layer.
    hidden_states = self.final_layernorm(hidden_states)
    # TENorm produces a "viewed" tensor. This will result in schedule.py's
    # deallocate_output_tensor() throwing an error, so a viewless tensor is
    # created to prevent this.
    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

    return hidden_states
