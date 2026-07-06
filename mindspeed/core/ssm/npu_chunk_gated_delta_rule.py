# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MindSpeed chunk_gated_delta_rule — Triton-accelerated with torch fallback.

Provides a ``chunk_gated_delta_rule`` function matching the FLA signature.
The forward path dispatches to MindSpeed's 7-kernel Triton pipeline; the
backward path uses a custom ``torch.autograd.Function`` for efficient
gradient computation.  Falls back to Megatron-LM's built-in
``torch_chunk_gated_delta_rule`` when Triton is unavailable or when an
``initial_state`` is provided.
"""

import warnings

import torch

from mindspeed.ops.triton.l2norm import l2norm_fwd, l2norm_bwd
from mindspeed.ops.triton.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from mindspeed.ops.triton.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from mindspeed.ops.triton.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from mindspeed.ops.triton.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from mindspeed.ops.triton.solve_tril import solve_tril
from mindspeed.ops.triton.cumsum import chunk_local_cumsum
from mindspeed.ops.triton.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


# =========================================================================
# Forward / backward pipelines
# =========================================================================


def _chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    chunk_size,
):
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=False)
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return g, o, A, final_state


def _chunk_gated_delta_rule_bwd(
    q,
    k,
    v,
    g,
    beta,
    A,
    scale,
    initial_state,
    do,
    dht,
    cu_seqlens,
    chunk_size,
):
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(f"dg current type is {dg.dtype}, should be float32")
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens, head_first=False)
    return dq, dk, dv, db, dg, dh0


# =========================================================================
# Autograd function
# =========================================================================


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, use_qk_l2norm_in_kernel, chunk_size
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g_cum, o, A, final_state = _chunk_gated_delta_rule_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            chunk_size,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_cum, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = _chunk_gated_delta_rule_bwd(
            q,
            k,
            v,
            g,
            beta,
            A,
            ctx.scale,
            initial_state,
            do,
            dht,
            cu_seqlens,
            ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None, None


# =========================================================================
# Validation
# =========================================================================


def _validate_inputs(q, k, v, g, beta, cu_seqlens, initial_state):
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype}, k current type is {k.dtype}, v current type is {v.dtype}, should be equal"
        )
    if q.dtype == torch.float32:
        raise ValueError("ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16.")
    if len(beta.shape) != 3:
        raise ValueError(
            f"beta current shape len is {len(beta.shape)}, "
            f"beta must be of shape [B, T, H] if head_first=False, "
            f"or [B, H, T] otherwise."
        )
    if q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests format mismatch: seq_len ({q.shape[1]}) "
            f"< num_heads ({q.shape[2]}). "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} "
                f"when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the "
                f"number of input sequences, i.e., {len(cu_seqlens) - 1} "
                f"rather than {initial_state.shape[0]}."
            )


# =========================================================================
# Public API
# =========================================================================


@torch.compiler.disable
def chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    chunk_size=64,
    head_first=False,
):
    """Chunked gated delta rule — Triton-accelerated with torch fallback.

    FLA-compatible signature.  Uses the MindSpeed 7-kernel Triton pipeline
    for forward/backward when ``initial_state is None``; falls back to
    Megatron-LM's ``torch_chunk_gated_delta_rule`` otherwise.

    Args:
        q: ``[B, T, H, K]``
        k: ``[B, T, H, K]``
        v: ``[B, T, H, V]``
        g: ``[B, T, H]`` (forget gate, in log space)
        beta: ``[B, T, H]``
        scale: default ``1/sqrt(K)``
        initial_state: ``[N, H, K, V]`` or ``None``
        output_final_state: return final state if True
        use_qk_l2norm_in_kernel: apply L2 norm internally if True
        cu_seqlens: ``[N+1]`` cumulative lengths or ``None``
        chunk_size: chunk size (default 64)
        head_first: deprecated

    Returns:
        ``(o, final_state)`` where *o* is ``[B, T, H, V]``.
    """
    _validate_inputs(q, k, v, g, beta, cu_seqlens, initial_state)

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if initial_state is None:
        o, final_state = ChunkGatedDeltaRuleFunction.apply(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            use_qk_l2norm_in_kernel,
            chunk_size,
        )
        return o, final_state

    # Fallback to Megatron's pure-torch implementation
    from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule

    return torch_chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
