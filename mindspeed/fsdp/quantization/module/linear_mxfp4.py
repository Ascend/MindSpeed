# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.


import torch
import triton
import triton.language as tl

# The only block size supported by the MX (microscaling) format.
MX_BLOCK_SIZE: tl.constexpr = 32

# Bias of the E8M0 scale exponent.
E8M0_EXPONENT_BIAS: tl.constexpr = 127

_MXFP4_BF16_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 8}),
    triton.Config({"BLOCKS_PER_PROGRAM": 16}),
    triton.Config({"BLOCKS_PER_PROGRAM": 32}),
    triton.Config({"BLOCKS_PER_PROGRAM": 64}),
    triton.Config({"BLOCKS_PER_PROGRAM": 128}),
]


@triton.jit
def _e2m1_to_bf16_value(nibble, scale_exp):
    sign = tl.where(((nibble >> 3) & 0x1) == 1, -1.0, 1.0)
    exp = ((nibble >> 1) & 0x3).to(tl.float32)
    man = (nibble & 0x1).to(tl.float32)
    mag = tl.where(
        exp == 0.0,
        0.5 * man,
        (1.0 + 0.5 * man) * tl.exp2(exp - 1.0),
    )
    scale = tl.exp2(scale_exp.to(tl.float32) - E8M0_EXPONENT_BIAS)
    return (sign * mag * scale).to(tl.bfloat16)


@triton.autotune(
    configs=_MXFP4_BF16_AUTOTUNE_CONFIGS,
    key=["K_PACKED", "NUM_BLOCKS"],
)
@triton.jit
def _mxfp4_to_bf16_kernel(
    x_ptr,  # *uint8, packed E2M1 pairs, shape [M, K_PACKED]
    scale_ptr,  # *uint8, E8M0 exponents, shape [M, K // MX_BLOCK]
    out_i32_ptr,  # *int32, shape [M, K_PACKED], each int32 stores two BF16 values
    K_PACKED,
    NUM_BLOCKS,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    MX_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_blk = tl.program_id(1)

    mx_blk = pid_blk * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    packed_in_mx_blk = tl.arange(0, MX_BLOCK // 2)
    packed_offsets = mx_blk[:, None] * (MX_BLOCK // 2) + packed_in_mx_blk[None, :]
    mask = packed_offsets < K_PACKED

    packed = tl.load(
        x_ptr + pid_m * K_PACKED + packed_offsets,
        mask=mask,
        other=0,
    ).to(tl.uint32)

    # One E8M0 scale is shared by 32 MXFP4 elements, i.e. 16 packed bytes.
    scale_exp = tl.load(
        scale_ptr + pid_m * NUM_BLOCKS + mx_blk,
        mask=mx_blk < NUM_BLOCKS,
        other=E8M0_EXPONENT_BIAS,
    )
    scale_exp = scale_exp[:, None]

    val_even = _e2m1_to_bf16_value(packed & 0xF, scale_exp)
    val_odd = _e2m1_to_bf16_value((packed >> 4) & 0xF, scale_exp)
    bits_even = val_even.to(tl.uint16, bitcast=True).to(tl.uint32)
    bits_odd = val_odd.to(tl.uint16, bitcast=True).to(tl.uint32)
    packed_bf16 = (bits_even & 0xFFFF) | ((bits_odd & 0xFFFF) << 16)

    # Store the two adjacent BF16 outputs as one contiguous 32-bit word.
    tl.store(
        out_i32_ptr + pid_m * K_PACKED + packed_offsets,
        packed_bf16.to(tl.int32, bitcast=True),
        mask=mask,
    )


def mxfp4_to_bf16_dequant(
    x_fp4: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantizes MXFP4 into BF16 with the optimized normal-range fast path.

    This benchmark helper assumes every non-zero scaled value stays in the
    normal BF16 exponent range. It does not handle BF16 subnormal or overflow
    cases and should not replace the general dequantization path.
    """
    if scale is None:
        raise ValueError("scale must be provided for MXFP4 to BF16 dequantization")

    if x_fp4.dtype != torch.uint8:
        x_fp4 = x_fp4.view(torch.uint8)
    if scale.dtype != torch.uint8:
        scale = scale.view(torch.uint8)

    orig_shape = x_fp4.shape
    x2d = x_fp4.reshape(-1, orig_shape[-1]).contiguous()
    m, k_packed = x2d.shape
    k = 2 * k_packed

    if k % MX_BLOCK_SIZE != 0:
        raise ValueError(f"last dim ({k} elements) must be divisible by the MX block size {MX_BLOCK_SIZE}")
    num_blocks = k // MX_BLOCK_SIZE
    scale2d = scale.reshape(m, num_blocks).contiguous()

    out_i32 = torch.empty((m, k_packed), dtype=torch.int32, device=x_fp4.device)
    grid = lambda meta: (m, triton.cdiv(num_blocks, meta["BLOCKS_PER_PROGRAM"]))  # pylint: disable=unnecessary-lambda-assignment  # noqa
    _mxfp4_to_bf16_kernel[grid](
        x_ptr=x2d,
        scale_ptr=scale2d,
        out_i32_ptr=out_i32,
        K_PACKED=k_packed,
        NUM_BLOCKS=num_blocks,
        MX_BLOCK=MX_BLOCK_SIZE,
    )

    return out_i32.view(torch.bfloat16).reshape(*orig_shape[:-1], k)
