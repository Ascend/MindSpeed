# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs


@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2]
    ],
    key=["H", "HV", "K", "BT", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    pid_t = tl.program_id(0)

    for i_bh in range(BH):
        i_b = i_bh // HV
        i_h = i_bh % HV

        if IS_VARLEN:
            i_n = tl.load(chunk_indices + pid_t * 2).to(tl.int32)
            i_tc = tl.load(chunk_indices + pid_t * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_cur = eos - bos
            chunk_off = i_tc
        else:
            bos = i_b * T
            T_cur = T
            chunk_off = pid_t

        o_t = chunk_off * BT + tl.arange(0, BT)
        m_t = o_t < T_cur

        p_b = tl.make_block_ptr(
            beta + bos * HV + i_h,
            (T_cur,),
            (HV,),
            (chunk_off * BT,),
            (BT,),
            (0,),
        )
        b_b = tl.load(p_b, boundary_check=(0,))

        b_A = tl.zeros([BT, BT], dtype=tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k + (bos * H + i_h // (HV // H)) * K,
                (T_cur, K),
                (H * K, 1),
                (chunk_off * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_A += tl.dot(b_k, tl.trans(b_k))

        if USE_G:
            p_g = tl.make_block_ptr(
                g + bos * HV + i_h,
                (T_cur,),
                (HV,),
                (chunk_off * BT,),
                (BT,),
                (0,),
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_diff = b_g[:, None] - b_g[None, :]
            b_A *= exp(b_g_diff)

        b_A *= b_b[:, None]

        m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0)

        p_A = tl.make_block_ptr(
            A + (bos * HV + i_h) * BT,
            (T_cur, BT),
            (BT * HV, 1),
            (chunk_off * BT, 0),
            (BT, BT),
            (1, 0),
        )
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, HV = *k.shape, beta.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    A = torch.empty(B, T, HV, BT, device=k.device, dtype=output_dtype)

    chunk_scaled_dot_kkt_fwd_kernel[(NT,)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BH=B * HV,
        sync_solver=True,
        multibuffer=True,
        limit_auto_multi_buffer_of_local_buffer="no-limit",
        set_workspace_multibuffer=2,
        enable_auto_bind_sub_block=True,
    )

    return A