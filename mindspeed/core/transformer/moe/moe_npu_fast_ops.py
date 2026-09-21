# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
"""NPU MoE fast ops and dispatch temporary lifetime optimization.

Megatron 0.15+ builds the router ``routing_probs`` / ``routing_map`` with
``index_put_`` and the MoE ``unpermute`` with ``index_add_`` when
``torch.use_deterministic_algorithms(True)`` is set, and Megatron 0.17+
replaced the ``masked_select`` based ``permute`` with an ``argsort`` based
implementation for CUDA graph compatibility. On NPU those selections lower to
slow AICPU kernels (``aclnnSort`` / ``aclnnIndexPutImpl``), while
``masked_select`` / ``Tensor.scatter`` / ``scatter_add_`` keep the fast
vector-core paths and provide the same determinism level that MindSpeed
master (mcore 0.12.x) runs with under ``--npu-deterministic``.
"""

from functools import wraps
from typing import Optional, Tuple

import torch

from megatron.core.transformer.moe.moe_utils import permute as megatron_permute
from megatron.core.transformer.moe.moe_utils import unpermute as megatron_unpermute


def fast_ops_enabled():
    # Full validation happens after patches are installed. Never cache the early
    # parser's default: all installed wrappers must honor the final switch.
    from mindspeed.args_utils import get_full_args

    return getattr(get_full_args(), "moe_npu_fast_ops", True)


def permute_npu_wrapper(fn):
    """Delegate unsupported calls to the implementation being wrapped."""

    # The module-level alias below intentionally holds an instance of this wrapper.
    @wraps(fn)
    def permute_npu(  # pylint: disable=redefined-outer-name
        tokens: torch.Tensor,
        routing_map: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        num_out_tokens: Optional[int] = None,
        fused: bool = False,
        drop_and_pad: bool = False,
        tokens_per_expert: Optional[torch.Tensor] = None,
        align_size: int = 0,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Permute tokens grouped by expert, restoring the Megatron 0.12 ``masked_select`` fast path.

        Keeps the Megatron 0.17+ five-value return contract. The dropless branch uses
        ``masked_select`` instead of ``argsort`` to avoid the AICPU ``aclnnSort`` kernel
        on NPU (CUDA graphs, the motivation for the upstream change, are not used in NPU
        training). Unsupported calls delegate to the implementation being wrapped.
        """
        if not fast_ops_enabled() or fused or drop_and_pad or align_size > 0:
            return fn(
                tokens,
                routing_map,
                probs=probs,
                num_out_tokens=num_out_tokens,
                fused=fused,
                drop_and_pad=drop_and_pad,
                tokens_per_expert=tokens_per_expert,
                align_size=align_size,
            )

        num_tokens, _ = tokens.shape
        num_experts = routing_map.shape[1]
        routing_map = routing_map.bool().T.contiguous()
        token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        sorted_indices = token_indices.masked_select(routing_map)

        # use the mapping to permute the tokens
        permuted_input = tokens.index_select(0, sorted_indices)

        if probs is not None:
            permuted_probs = probs.T.contiguous().masked_select(routing_map)
        else:
            permuted_probs = None

        return permuted_input, permuted_probs, sorted_indices, None, tokens_per_expert

    return permute_npu


def unpermute_npu_wrapper(fn):
    """Delegate unsupported calls to the implementation being wrapped."""

    # The module-level alias below intentionally holds an instance of this wrapper.
    @wraps(fn)
    def unpermute_npu(  # pylint: disable=redefined-outer-name
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        restore_shape: torch.Size,
        probs: Optional[torch.Tensor] = None,
        routing_map: Optional[torch.Tensor] = None,
        fused: bool = False,
        drop_and_pad: bool = False,
        pad_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Restore the original token order, using ``scatter_add_`` on the supported fast path.

        Megatron 0.15+ selects ``index_add_`` under deterministic algorithms, which lowers
        to a slower NPU kernel than ``aclnnScatterAdd``. Behavior otherwise matches
        Megatron 0.18 (and the op selection matches MindSpeed master / mcore 0.12).
        """
        if not fast_ops_enabled() or fused or drop_and_pad or pad_offsets is not None:
            return fn(
                permuted_tokens,
                sorted_indices,
                restore_shape,
                probs=probs,
                routing_map=routing_map,
                fused=fused,
                drop_and_pad=drop_and_pad,
                pad_offsets=pad_offsets,
            )

        _, hidden = restore_shape
        input_dtype = permuted_tokens.dtype

        if probs is not None:
            assert routing_map is not None, "Mask must be provided to permute the probs."
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
            permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

        # Create an output tensor filled with zeros and scatter add the permuted
        # input back to the original positions.
        output_tokens = torch.zeros(restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
        output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
        return output_tokens.to(dtype=input_dtype)

    return unpermute_npu


def topk_routing_with_score_function_scatter_wrapper(fn):
    """Build the router routing tensors with ``Tensor.scatter`` instead of deterministic ``index_put_``.

    Megatron 0.15+ constructs ``routing_probs`` / ``routing_map`` via ``index_put_``
    when deterministic algorithms are enabled, which lowers to the AICPU
    ``aclnnIndexPutImpl`` kernel on NPU. The wrapped function is called with
    ``dense_output=True`` (bypassing that branch) and the sparse tensors are rebuilt
    with ``scatter``, matching the Megatron 0.12 op selection.
    """

    @wraps(fn)
    def wrapper(
        logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool = False,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        scaling_factor: Optional[float] = None,
        score_function: str = "softmax",
        expert_bias: Optional[torch.Tensor] = None,
        fused: bool = False,
        router_replay: Optional[object] = None,
        dense_output: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not fast_ops_enabled() or fused:
            return fn(
                logits,
                topk,
                use_pre_softmax=use_pre_softmax,
                num_groups=num_groups,
                group_topk=group_topk,
                scaling_factor=scaling_factor,
                score_function=score_function,
                expert_bias=expert_bias,
                fused=fused,
                router_replay=router_replay,
                dense_output=dense_output,
                **kwargs,
            )

        probs, top_indices = fn(
            logits,
            topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=expert_bias,
            fused=fused,
            router_replay=router_replay,
            dense_output=True,
            **kwargs,
        )
        if dense_output:
            return probs, top_indices

        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
        return routing_probs, routing_map

    return wrapper


def moe_forward_dispatch_lifetime_wrapper(forward):
    """Keep native staged APIs while shortening the full eager forward's locals.

    Megatron's staged forward retains both permutation-1 and EP receive tensors
    in its outer frame while experts run. Rebind those operands after each
    consumer. Autograd and collective backends remain responsible for storage
    they still need; this wrapper never resizes, detaches, or pools storage.
    """
    from megatron.core.transformer.moe import moe_layer as mcore

    native_compute = getattr(mcore.MoELayer, "routed_experts_compute", None)
    if native_compute is None:
        return forward
    record_dgrad = getattr(mcore, "_RecordExpertDgradCompletion", None)
    execution_map = {"route", "expert_compute", "postprocess"}

    @wraps(forward)
    def wrapper(self, hidden_states, intermediate_tensors=None, padding_mask=None):
        # Keep all native-stage requirements in one short-circuiting fallback guard.
        if (
            not fast_ops_enabled()  # pylint: disable=too-many-boolean-expressions
            or not self.training
            or intermediate_tensors is not None
            or self.config.cuda_graph_impl != "none"
            # Megatron stores a list; compare stage membership, as native forward does.
            or set(self.fwd_execution_map) != execution_map
            or hasattr(self, "_inference_token_dispatcher")
            or getattr(self.routed_experts_compute, "__func__", None) is not native_compute
            or (self.config.overlap_dispatch_backward_with_experts_wgrad and record_dgrad is None)
        ):
            return forward(self, hidden_states, intermediate_tensors, padding_mask)

        if self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()

        def custom_forward(hidden_states, intermediate_tensors=None, padding_mask=None):
            shared_expert_output = self.shared_experts_compute(hidden_states)
            probs, routing_map = self.route(hidden_states, padding_mask)
            hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)

            # Drop the outer reference to permutation-1 before expert execution.
            hidden_states, probs = self.dispatch(hidden_states, probs)
            if self.config.overlap_dispatch_backward_with_experts_wgrad:
                hidden_states = record_dgrad.apply(self._delayed_wgrad_event, hidden_states)
            # Drop the EP receive tensor if TP gather/local sorting replaces it.
            # If postprocessing aliases it, its actual consumers retain it.
            hidden_states, tokens_per_expert, probs = self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
            hidden_states, mlp_bias = mcore.apply_module(self.experts)(hidden_states, tokens_per_expert, probs)
            assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
            hidden_states = self.token_dispatcher.combine_preprocess(hidden_states)
            hidden_states = self.combine(hidden_states)
            hidden_states = self.postprocess(hidden_states, shared_expert_output)
            return hidden_states, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8 or self.config.fp4:
                return mcore.te_checkpoint(
                    custom_forward,
                    False,
                    mcore.tensor_parallel.random.get_cuda_rng_tracker,
                    self.tp_group,
                    hidden_states,
                    intermediate_tensors,
                    padding_mask,
                )
            return mcore.tensor_parallel.checkpoint(
                custom_forward, False, hidden_states, intermediate_tensors, padding_mask
            )
        return custom_forward(hidden_states, intermediate_tensors, padding_mask)

    return wrapper


permute_npu = permute_npu_wrapper(megatron_permute)
unpermute_npu = unpermute_npu_wrapper(megatron_unpermute)
