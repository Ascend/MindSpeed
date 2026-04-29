# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import os
from typing import List, Optional, Tuple

import pytest
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

from mindspeed.ops.triton.utils import assert_close
from mindspeed.ops.triton.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
from mindspeed.ops.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from mindspeed.ops.triton.cumsum import chunk_local_cumsum
from mindspeed.ops.triton.chunk_o import chunk_bwd_dv_local
from mindspeed.ops.triton.wy_fast import recompute_w_u_fwd
from mindspeed.ops.triton.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from mindspeed.ops.triton.solve_tril import solve_tril
from mindspeed.ops.triton.utils import prepare_chunk_indices, prepare_chunk_offsets


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_ref(
    q,
    k,
    w,
    g,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    dh += (boh * H + i_h) * K * V
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K
    if USE_INITIAL_STATE:
        dh0 += i_nh * K * V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K * V

    if USE_FINAL_STATE_GRADIENT:
        p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if K > 64:
            p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if K > 128:
            p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 192:
            p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        p_dh1 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            bg_last = tl.load(g + (bos + last_idx) * H + i_h)
            bg_last_exp = tl.exp(bg_last)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_exp = tl.exp(b_g)
        else:
            bg_last = None
            last_idx = None
            b_g = None
            b_g_exp = None

        p_dv = tl.make_block_ptr(dv, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)

        # Update dv
        p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_dv *= tl.where(m_t, tl.exp(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        # Update dh
        p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        b_q = (b_q * scale).to(b_q.dtype)
        b_dh1 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh2 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh3 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh4 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_bwd_dhu_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    dht: Optional[torch.Tensor],
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    BV = 64
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_ref[(triton.cdiv(V, BV), N * H)](
        q=q,
        k=k,
        w=w,
        g=g,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
    )
    return dh, dh0, dv2


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'chunk_size', 'cu_seqlens'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-chunk_size{}-cu_seqlens{}".format(*test))
        for test in [
        (1, 1024, 32, 128, 64, None),
        (2, 1024, 32, 128, 64, None),
        (1, 32768, 32, 128, 64, None),
        (1, 1024, 32, 128, 64, [0, 10, 66, 140, 229, 351, 401, 574, 684, 819, 874, 922, 1024]),
        (1, 32768, 32, 128, 64, [0, 10, 66, 140, 229, 351, 401, 574, 684, 819,
        874, 922, 1069, 1141, 1244, 1326, 1453, 1564, 1616, 1764,
        1894, 1972, 2147, 2224, 2294, 2362, 2452, 2522, 2684, 2841,
        2927, 3096, 3161, 3273, 3373, 3441, 3482, 3595, 3652, 3735,
        3830, 3873, 3986, 4075, 4232, 4372, 4430, 4523, 4571, 4672,
        4793, 4990, 5053, 5115, 5269, 5331, 5472, 5609, 5668, 5783,
        5962, 6088, 6175, 6275, 6331, 6460, 6710, 6859, 6905, 7073,
        7221, 7320, 7434, 7595, 7646, 7691, 7769, 7880, 7976, 8039,
        8106, 8519, 8708, 8766, 8851, 9162, 9213, 9302, 9385, 9502,
        9566, 9624, 9770, 9815, 9904, 9972, 10056, 10186, 10270, 10366,
        10457, 10528, 10612, 10711, 10818, 10943, 10993, 11082, 11157, 11232,
        11354, 11454, 11587, 11693, 11772, 11859, 11927, 11976, 12067, 12196,
        12250, 12313, 12400, 12495, 12691, 12754, 12953, 13069, 13162, 13221,
        13314, 13461, 13701, 13752, 13877, 14052, 14201, 14347, 14414, 14492,
        14640, 14760, 14837, 14919, 15049, 15154, 15264, 15373, 15551, 15672,
        15791, 15859, 15906, 15964, 16119, 16189, 16280, 16333, 16416, 16482,
        16527, 16597, 16647, 16719, 16828, 16913, 16976, 17145, 17197, 17332,
        17561, 17880, 18090, 18141, 18269, 18413, 18471, 18559, 18631, 18700,
        18788, 18885, 19043, 19093, 19137, 19278, 19349, 19465, 19632, 19702,
        19773, 19839, 19896, 19989, 20146, 20209, 20286, 20336, 20427, 20505,
        20573, 20628, 20689, 20767, 20929, 21002, 21124, 21234, 21296, 21409,
        21492, 21553, 21626, 21712, 21865, 21992, 22106, 22177, 22244, 22378,
        22449, 22519, 22588, 22702, 22752, 22844, 22942, 23036, 23243, 23382,
        23445, 23519, 23661, 23756, 23832, 24364, 24422, 24487, 24809, 24870,
        25101, 25206, 25562, 25623, 25659, 25764, 25861, 26015, 26187, 26250,
        26298, 26578, 26680, 26792, 26950, 27012, 27060, 27173, 27228, 27427,
        27504, 27595, 27692, 27753, 27807, 27931, 27982, 28104, 28166, 28222,
        28410, 28548, 28597, 28716, 28773, 28928, 29088, 29363, 29502, 29641,
        29700, 29756, 30026, 30213, 30285, 30430, 30586, 30694, 30849, 30946,
        31047, 31110, 31163, 31263, 31326, 31365, 31455, 31527, 31738, 31851,
        31920, 32003, 32070, 32173, 32223, 32258, 32383, 32443, 32617, 32768]),
    ]
    ]
)
def test_chunk_gated_delta_rule_bwd_dhu(B, T, H, D, chunk_size, cu_seqlens):
    device = "npu:0"
    torch.manual_seed(42)
    torch.npu.manual_seed(42)

    data_type = torch.bfloat16
    chunk_size = chunk_size
    K = D
    V = K
    q = torch.randn(B, T, H, K, dtype=data_type).npu()
    k = F.normalize(torch.randn(B, T, H, K, dtype=data_type).npu(), p=2, dim=-1)
    v = torch.randn(B, T, H, V, dtype=data_type).npu()
    beta = torch.rand(B, T, H, dtype=data_type).npu().sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=data_type)).npu()
    initial_state = None
    if cu_seqlens is not None:
        cu_seqlens = torch.LongTensor(cu_seqlens).to('npu:0')
    scale = 1. / (K ** 0.5)
    output_final_state = False
    dht = None
    do = torch.randn(B, T, H, V, dtype=data_type).npu()

    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=True)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=torch.float32
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype
    )
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
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
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

    _, _, ref_dv2 = chunk_gated_delta_rule_bwd_dhu_ref(
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

    _, _, dv2 = chunk_gated_delta_rule_bwd_dhu(
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

    print("dv diff:", torch.max(torch.abs(ref_dv2 - dv2)))
    assert_close('dv', ref_dv2, dv2, 0.001)