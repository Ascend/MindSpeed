"""FLA-NPU AscendC/Triton GDR adapter for GatedDeltaNet.

The public API uses sequence-first ``[B, T, H, D]``. FLA-NPU cumsum retains
sequence-first gates, while KKT and AscendC tensor inputs use head-first
``[B, H, T, D]`` tensors.
"""

import warnings

import torch

from fla_npu.ops.ascendc import (
    npu_chunk_bwd_dqkwg,
    npu_chunk_bwd_dv_local,
    npu_chunk_fwd_o,
    npu_chunk_gated_delta_rule_bwd_dhu,
    npu_chunk_gated_delta_rule_fwd_h,
    npu_prepare_wy_repr_bwd_da,
    npu_prepare_wy_repr_bwd_full,
    npu_recompute_w_u_fwd,
    npu_solve_tri,
)
from fla_npu.ops.triton import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
    input_guard,
    l2norm_bwd,
    l2norm_fwd,
)


def _as_int_list(cu_seqlens):
    return [int(value) for value in cu_seqlens.detach().cpu().tolist()]


def _next_power_of_2(value):
    return 1 << (value - 1).bit_length()


def _chunk_index_pairs(cu_seqlens_list, chunk_size):
    pairs = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens_list, cu_seqlens_list[1:])):
        for chunk_idx in range((end - start + chunk_size - 1) // chunk_size):
            pairs.append((seq_idx, chunk_idx))
    return pairs


def _prepare_packed_metadata(cu_seqlens, g, chunk_size):
    if cu_seqlens is None:
        return None, None, None, None
    cu_seqlens = cu_seqlens.to(device=g.device, dtype=torch.int64)
    cu_list = _as_int_list(cu_seqlens)
    chunk_pairs = _chunk_index_pairs(cu_list, chunk_size)
    chunk_indices = [value for pair in chunk_pairs for value in pair]
    cumsum_block_t = _next_power_of_2((1 << 17) // (g.shape[-1] * chunk_size))
    chunk_indices_out = {}
    for size in dict.fromkeys((chunk_size, cumsum_block_t)):
        pairs = chunk_pairs if size == chunk_size else _chunk_index_pairs(cu_list, size)
        chunk_indices_out[str(size)] = torch.tensor(pairs, device=cu_seqlens.device, dtype=torch.int64).reshape(-1, 2)
    return cu_seqlens, cu_list, chunk_indices, chunk_indices_out


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
    cu_seqlens_list,
    chunk_indices,
    chunk_indices_out,
    chunk_size,
):
    g_cum = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices_out,
        head_first=False,
    )
    k_h = k.transpose(1, 2).contiguous()
    A = chunk_scaled_dot_kkt_fwd(
        k=k_h,
        g=g_cum,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=(chunk_indices_out[str(chunk_size)] if chunk_indices_out is not None else None),
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    if cu_seqlens_list is None:
        A = npu_solve_tri(A.to(k.dtype).contiguous(), layout="bsnd")
    else:
        A = npu_solve_tri(
            A.squeeze(0).to(k.dtype).contiguous(),
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_indices,
            layout="tnd",
        ).unsqueeze(0)

    q_h = q.transpose(1, 2).contiguous()
    v_h = v.transpose(1, 2).contiguous()
    g_h = g_cum.transpose(1, 2).contiguous()
    beta_h = beta.transpose(1, 2).contiguous().float()
    A_h = A.transpose(1, 2).contiguous()
    w, u = npu_recompute_w_u_fwd(
        k_h,
        v_h,
        beta_h,
        A_h,
        chunk_size,
        g=g_h,
        gk=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = npu_chunk_gated_delta_rule_fwd_h(
        k_h,
        w,
        u,
        g=g_h,
        gk=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
    )
    if not output_final_state:
        final_state = None
    o_h = npu_chunk_fwd_o(
        q_h,
        k_h,
        v_new,
        h,
        scale,
        g=g_h,
        g_gamma=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )
    return g_cum, o_h.transpose(1, 2).contiguous(), A_h, final_state


def _chunk_gated_delta_rule_bwd(
    q,
    k,
    v,
    g,
    beta,
    A,
    scale,
    _initial_state,
    do,
    _dht,
    cu_seqlens,
    cu_seqlens_list,
    chunk_indices,
    chunk_indices_out,
    chunk_size,
):
    q_h = q.transpose(1, 2).contiguous()
    k_h = k.transpose(1, 2).contiguous()
    v_h = v.transpose(1, 2).contiguous()
    g_h = g.transpose(1, 2).contiguous()
    beta_h = beta.transpose(1, 2).contiguous().float()
    w, u = npu_recompute_w_u_fwd(
        k_h,
        v_h,
        beta_h,
        A,
        chunk_size,
        g=g_h,
        gk=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
    )
    h, v_new, _ = npu_chunk_gated_delta_rule_fwd_h(
        k_h,
        w,
        u,
        g=g_h,
        gk=None,
        initial_state=None,
        output_final_state=False,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
    )
    do_h = do.transpose(1, 2).contiguous()
    dv = npu_chunk_bwd_dv_local(
        q_h,
        k_h,
        do_h,
        g_h,
        scale,
        chunk_size,
        g_gamma=None,
        A=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
    )
    dh, _, dv = npu_chunk_gated_delta_rule_bwd_dhu(
        q_h,
        k_h,
        w,
        do_h,
        dv,
        scale,
        chunk_size,
        g=g_h,
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
        use_exp2=False,
        transpose_state_layout=False,
    )
    dq_h, dk_h, dw, dg_h = npu_chunk_bwd_dqkwg(
        q_h,
        k_h,
        v_new,
        g_h,
        h,
        do_h,
        dh,
        dv,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
        w=None,
        g_gamma=None,
        scale=scale,
        use_exp2=False,
        transpose_state_layout=False,
    )
    dA = npu_prepare_wy_repr_bwd_da(
        k_h,
        v_h,
        beta_h.float(),
        A,
        dw,
        dv,
        g_h.float(),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
    )
    dk2_h, dv_h, dbeta_h, dg2_h = npu_prepare_wy_repr_bwd_full(
        k_h,
        v_h,
        beta_h,
        A,
        dA,
        dw,
        dv,
        g_h,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_indices,
    )
    dk_h = dk_h + dk2_h
    dg = dg_h.transpose(1, 2).contiguous() + dg2_h.transpose(1, 2).contiguous()
    if dg.dtype != torch.float32:
        raise ValueError(f"dg current type is {dg.dtype}, should be float32")
    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices_out,
        head_first=False,
    )
    return (
        dq_h.transpose(1, 2).contiguous(),
        dk_h.transpose(1, 2).contiguous(),
        dv_h.transpose(1, 2).contiguous(),
        dg,
        dbeta_h.transpose(1, 2).contiguous(),
    )


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
        cu_seqlens, cu_seqlens_list, chunk_indices, chunk_indices_out = _prepare_packed_metadata(
            cu_seqlens, g, chunk_size
        )
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
            cu_seqlens_list,
            chunk_indices,
            chunk_indices_out,
            chunk_size,
        )
        ctx.save_for_backward(q, k, v, g_cum, beta, A)
        ctx.q_rstd = q_rstd
        ctx.k_rstd = k_rstd
        ctx.initial_state = initial_state
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.chunk_indices_out = chunk_indices_out
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, g, beta, A = ctx.saved_tensors
        dq, dk, dv, dg, dbeta = _chunk_gated_delta_rule_bwd(
            q,
            k,
            v,
            g,
            beta,
            A,
            ctx.scale,
            ctx.initial_state,
            do,
            dht,
            ctx.cu_seqlens,
            ctx.cu_seqlens_list,
            ctx.chunk_indices,
            ctx.chunk_indices_out,
            ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, ctx.q_rstd, dq)
            dk = l2norm_bwd(k, ctx.k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), dbeta.to(beta), None, None, None, None, None, None


def _validate_inputs(q, k, v, g, beta, cu_seqlens, initial_state, output_final_state):
    if initial_state is not None:
        raise NotImplementedError("initial_state is not supported by the AscendC GDR adapter")
    if output_final_state:
        raise NotImplementedError("output_final_state is not supported by the AscendC GDR adapter")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype}, k current type is {k.dtype}, v current type is {v.dtype}, should be equal"
        )
    if q.dtype == torch.float32:
        raise ValueError("ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16.")
    if len(beta.shape) != 3:
        raise ValueError(
            f"beta current shape len is {len(beta.shape)}, "
            "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
        )
    if q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                "Please flatten variable-length inputs before processing."
            )


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
    """Run GDR with BTHD public tensors and AscendC raw main operators."""
    if head_first:
        raise ValueError("Only head_first=False is supported.")
    _validate_inputs(q, k, v, g, beta, cu_seqlens, initial_state, output_final_state)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return ChunkGatedDeltaRuleFunction.apply(
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
