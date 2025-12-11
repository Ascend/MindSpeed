# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025; Songlin Yang; Yu Zhang
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

import pytest
import torch
import triton
import triton.language as tl

from mindspeed.lite.ops.triton.l2norm import l2norm_bwd
from mindspeed.lite.ops.triton.utils import assert_close
from mindspeed.lite.ops.triton.utils import is_amd

BT_LIST = [8, 16, 32, 64, 128]
NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if is_amd else [1, 2, 4, 8, 16, 32]


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D']
)
@triton.jit
def l2norm_bwd_kernel1(
    y,
    rstd,
    dy,
    dx,
    eps,
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    y += i_t * D
    dx += i_t * D
    dy += i_t * D

    cols = tl.arange(0, BD)
    mask = cols < D
    b_y = tl.load(y + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + i_t).to(tl.float32)
    b_dy = tl.load(dy + cols, mask=mask, other=0.0).to(tl.float32)
    b_dx = b_dy * b_rstd - tl.sum(b_dy * b_y) * b_y * b_rstd
    tl.store(dx + cols, b_dx, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in BT_LIST
    ],
    key=['D', 'NB']
)
@triton.jit
def l2norm_bwd_kernel(
    y,
    rstd,
    dy,
    dx,
    eps,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_dy = tl.make_block_ptr(dy, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_dx = tl.make_block_ptr(dx, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))

    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = tl.load(p_rstd, boundary_check=(0,)).to(tl.float32)
    b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
    b_dx = b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


def ref_l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6
):
    y_shape_og = y.shape
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    assert dy.shape == y.shape
    # allocate output
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // y.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return (triton.cdiv(T, meta['BT']), )

        l2norm_bwd_kernel[grid](
            y=y,
            rstd=rstd,
            dy=dy,
            dx=dx,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            NB=NB,
        )
    else:
        l2norm_bwd_kernel1[(T,)](
            y=y,
            rstd=rstd,
            dy=dy,
            dx=dx,
            eps=eps,
            D=D,
            BD=BD,
        )

    return dx.view(y_shape_og)


@pytest.mark.skip(reason='Hanged to be fixed')
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}".format(*test))
        for test in [
            (1, 1024, 32, 128),
            (1, 4096, 32, 128),
        ]
    ]
)
def test_l2norm_bwd(B, T, H, K):

    device = "npu:0"
    device_dtype = torch.float32

    y = torch.rand((B, T, H, K), device=device, dtype=device_dtype)
    rstd = torch.rand((B, T, H), device=device, dtype=device_dtype)
    dy = torch.rand((B, T, H, K), device=device, dtype=device_dtype)

    ref_dx = ref_l2norm_bwd(
            y=y,
            rstd=rstd,
            dy=dy,
        )
    
    dx = l2norm_bwd(
            y=y,
            rstd=rstd,
            dy=dy,
        )

    assert_close('dx', ref_dx, dx, 0.001)
