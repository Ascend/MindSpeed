# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import torch
from torch.autograd import Function

FP32_EXPONENT_BIAS = 127.0
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
# E4M3 max finite value: 2^8 * (1 + 6 / 8) = 448.
FP8_E4M3_MAX = 448.0


def _e4m3fn_fake_quantize(tensor):
    tensor = tensor.to(torch.float32)
    tensor_abs = tensor.abs()
    tensor_sign = torch.sign(tensor)

    private_exp = torch.floor(torch.log2(tensor_abs))
    private_exp = torch.maximum(private_exp, torch.full_like(private_exp, -6.0))
    quant_step = 2 ** (private_exp - 3.0)
    quantized = torch.round(tensor_abs / quant_step) * quant_step
    quantized = torch.clamp(quantized, 0.0, FP8_E4M3_MAX)
    quantized = tensor_sign * quantized

    return torch.where(torch.isnan(tensor), tensor, quantized)


def w8a16_fake_quant(tensor, ebits, mbits, qdim=-1):
    """Fake Quantization Function

    Args:
        tensor (torch.Tensor required): The original tensor to be fake quantized.
        ebits (int required): Reserved for API consistency with W4A16 fake quantization.
        mbits (int required): Reserved for API consistency with W4A16 fake quantization.
        qdim  (int optional): Dimension along which quantization is applied. Default is -1, indicating the last dim.

    Return:
        output (torch.Tensor): The tensor after fake quantization.
    """
    dim_size = tensor.shape[qdim]
    pad_size = (32 - dim_size % 32) % 32
    if pad_size != 0:
        pad_shape = list(tensor.shape)
        pad_shape[qdim] = pad_size
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=qdim)

    tensor = tensor.unflatten(qdim, (-1, 32))
    shared_exp = torch.amax(tensor.abs(), dim=qdim, keepdim=True)
    mask = (shared_exp == 0).float()
    safe_amax = shared_exp + FP32_MIN_NORMAL * mask
    max_norm = torch.tensor(FP8_E4M3_MAX, dtype=tensor.dtype, device=tensor.device)
    shared_exp = torch.ceil(torch.log2(safe_amax) - torch.log2(max_norm))
    shared_exp = torch.where(mask.bool(), torch.zeros_like(shared_exp), shared_exp)
    scale_emax = 2 ** (8.0 - 1.0) - 1

    shared_exp = torch.where(shared_exp > scale_emax, torch.full_like(shared_exp, float('nan')), shared_exp)
    shared_exp = torch.where(shared_exp < -scale_emax, torch.full_like(shared_exp, -scale_emax), shared_exp)
    tensor = tensor / (2**shared_exp)
    tensor = torch.clamp(tensor, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    tensor = _e4m3fn_fake_quantize(tensor)
    recovered_tensor = tensor * (2**shared_exp)

    recovered_tensor = recovered_tensor.flatten(qdim - 1, qdim)
    if pad_size != 0:
        recovered_tensor = recovered_tensor.narrow(qdim, 0, dim_size)

    return recovered_tensor


class W8A16FakeQuantization(Function):
    @staticmethod
    def forward(ctx, high_precision_tensor, block_size, transpose):
        dequant_tensor = w8a16_fake_quant(high_precision_tensor, 4.0, 5.0, qdim=-1)
        return dequant_tensor.to(high_precision_tensor.dtype)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad, None, None
