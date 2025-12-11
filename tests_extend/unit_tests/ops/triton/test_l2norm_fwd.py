# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025; Songlin Yang; Yu Zhang
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import Optional

import pytest
import torch
import triton
import triton.language as tl

from mindspeed.lite.ops.triton.l2norm import l2norm_fwd
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
def l2norm_fwd_kernel1(
    x,
    y,
    rstd,
    eps,
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D

    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)
    tl.store(rstd + i_t, b_rstd)


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in BT_LIST
    ],
    key=['D', 'NB']
)
@triton.jit
def l2norm_fwd_kernel(
    x,
    y,
    rstd,
    eps,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))

    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))


def ref_l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return (triton.cdiv(T, meta['BT']), )

        l2norm_fwd_kernel[grid](
            x=x,
            y=y,
            rstd=rstd,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            NB=NB,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x=x,
            y=y,
            rstd=rstd,
            eps=eps,
            D=D,
            BD=BD,
        )
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


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
def test_l2norm_fwd(B, T, H, K):

    device = "npu:0"
    device_dtype = torch.float32

    x = torch.rand((B, T, H, K), device=device, dtype=device_dtype)

    ref_y, ref_rstd = ref_l2norm_fwd(
                    x=x
                )
    
    y, rstd = l2norm_fwd(
                    x=x
                )

    assert_close('y', ref_y, y, 0.001)
    assert_close('rstd', ref_rstd, rstd, 0.001)