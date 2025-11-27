# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

import os
from typing import List, Optional, Tuple

import pytest
import torch
import triton
import triton.language as tl

from mindspeed.lite.ops.triton.utils import prepare_chunk_indices, exp, check_shared_mem, assert_close
from mindspeed.lite.ops.triton.chunk_o import bwd_chunk_dqkwg


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'USE_DW': lambda args: args['dw'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_DW'],
)
@triton.jit(do_not_specialize=['T'])
def ref_chunk_bwd_kernel_dqkwg(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    do,
    dh,
    dq,
    dk,
    dg,
    w,
    dv,
    dw,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_DW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        all_T = T
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
        all_T = B * T

    # offset calculation
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V
    dh += (i_tg * H + i_h).to(tl.int64) * K * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K

    # for delta rule only
    if USE_DW:
        w += (bos * H + i_h) * K
        dw += (bos * H + i_h) * K
        dv += (bos * H + i_h) * V

    if USE_G:
        dg += i_k * all_T * H
        b_dg_last = tl.zeros([1, ], dtype=tl.float32) if USE_G else None
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(BT, T - i_t * BT)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        if USE_G:
            b_dg_last += (tl.sum(b_h * b_dh))
        # [BT, BV] @ [BV, BT] -> [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        if USE_DW:
            p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

    if USE_DW:
        p_dw = tl.make_block_ptr(dw, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    if USE_G:
        b_dg = tl.zeros([BT, ], dtype=tl.float32)
        g += bos * H + i_h
        dg += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
        b_dg_last *= exp(b_g_last)

        b_dq = b_dq * exp(b_g)[:, None] * scale
        b_dg += tl.sum(b_dq * b_q, axis=1)

        b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
        b_dg -= tl.sum(b_k * b_dk, axis=1)
        b_dg_last += tl.sum(b_dk * b_k)

        b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0) * scale
        b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))
        b_dg += tl.sum(b_ds2, axis=1)
        b_dg -= tl.sum(b_ds2, axis=0)

        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

    elif USE_G_GAMMA:
        b_dq = b_dq * exp(b_g)[:, None] * scale
        b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
        b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0) * scale
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    else:
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


def ref_chunk_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
    dw = torch.empty_like(w) if w is not None else None

    grid = (NK, NT, B * H)
    ref_chunk_bwd_kernel_dqkwg[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dh=dh,
        dv=dv,
        w=w,
        dw=dw,
        dq=dq,
        dk=dk,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )

    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw, dg


@pytest.mark.skip(reason='Hanged to be fixed')
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'hidden_size', 'scale', 'chunk_size'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-hidden_size{}-scale{}-chunk_size{}".format(*test))
        for test in [
            (1, 1024, 32, 128, 2048, 0.5, 16),
            (1, 4096, 32, 128, 2048, 0.5, 16),
        ]
    ]
)
def test_chunk_bwd_dqkwg(B, T, H, D, hidden_size, scale, chunk_size):

    device = "npu:0"
    device_dtype = torch.float32

    q = torch.rand((B, T, H, D), device=device, dtype=device_dtype)
    k = torch.rand((B, T, H, D), device=device, dtype=device_dtype)
    v = torch.rand((B, T, H, D), device=device, dtype=device_dtype)
    w = torch.rand((B, T, H, D), device=device, dtype=device_dtype)
    g = torch.rand((B, T, H), device=device, dtype=device_dtype)
    h = torch.rand((B, hidden_size, H, D, D), device=device, dtype=device_dtype)
    dv = torch.rand((B, T, H, D), device=device, dtype=device_dtype)
    do = torch.rand((B, T, H, D), device=device, dtype=device_dtype)
    dh = torch.rand((B, hidden_size, H, D, D), device=device, dtype=device_dtype)

    ref_dq, ref_dk, ref_dw, ref_dg = ref_chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        do=do,
        dh=dh,
        dv=dv,
        w=w,
        scale=scale,
        chunk_size=chunk_size
    )
    
    dq, dk, dw, dg = bwd_chunk_dqkwg(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        do=do,
        dh=dh,
        dv=dv,
        w=w,
        scale=scale,
        chunk_size=chunk_size
    )

    assert_close('dq', ref_dq, dq, 0.001)
    assert_close('dk', ref_dk, dk, 0.001)
    assert_close('dw', ref_dw, dw, 0.001)
    assert_close('dg', ref_dg, dg, 0.001)


