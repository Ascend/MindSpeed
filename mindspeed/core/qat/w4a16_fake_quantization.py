import os

import torch
import torch_npu
from torch.autograd import Function

FP32_EXPONENT_BIAS = 127.0
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


def w4a16_fake_quant(tensor, ebits, mbits, qdim=-1):
    """Fake Quantization Function

        Args:
            tensor (torch.Tensor required): The original tensor to be fake quantized.
            ebits (int required): Number of bits assigned to the exponent in the quantization format.
            mbits (int required): Number of bits assigned to the mantissa in the quantization format.
            qdim  (int optional): Dimension along which quantization is applied. Default is -1, indicating the last dim.

        Return:
            output (torch.Tensor): The tensor after fake quantization.
    """
    emax = 2 ** (ebits - 1)
    max_norm = 2 ** emax * (2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)
    tensor = tensor.unflatten(qdim, (-1, 32))
    shared_exp = torch.amax(tensor.abs(), dim=qdim, keepdim=True)
    mask = (shared_exp == 0).float()
    shared_exp = torch.floor(torch.log2(shared_exp + FP32_MIN_NORMAL * mask))
    mask = (tensor > -FP32_EXPONENT_BIAS).float()
    tensor = tensor * mask
    shared_exp = shared_exp - emax
    scale_emax = 2 ** (8.0 - 1.0) - 1

    shared_exp = torch.where(shared_exp > scale_emax, torch.full_like(shared_exp, float('nan')), shared_exp)
    shared_exp = torch.where(shared_exp < -scale_emax, torch.full_like(shared_exp, -scale_emax), shared_exp)
    tensor = tensor / (2 ** shared_exp)
    mask = (tensor == 0).float()
    private_exp = torch.floor(torch.log2(tensor.abs() + mask))

    min_exp = -(2 ** (ebits - 1)) + 2
    private_exp = torch.maximum(private_exp, torch.tensor(min_exp, device=tensor.device))
    tensor = tensor / (2 ** private_exp) * (2 ** (mbits - 2))
    tensor_sign = torch.sign(tensor)
    tensor = tensor_sign * torch.floor(tensor.abs() + 0.5)
    tensor = tensor / (2 ** (mbits - 2)) * (2 ** private_exp)

    tensor = torch.clamp(tensor, -max_norm, max_norm)

    tensor = torch.where(torch.isinf(tensor), tensor, tensor)
    tensor = torch.where(torch.isnan(tensor), tensor, tensor)
    recovered_tensor = tensor * (2 ** shared_exp)

    recovered_tensor = recovered_tensor.flatten(qdim - 1, qdim)

    return recovered_tensor


class W4A16FakeQuantization(Function):
    @staticmethod
    def forward(ctx, fp32_tensor, block_size, transpose):
        ebits, mbits = 2.0, 3.0
        dequant_tensor = w4a16_fake_quant(fp32_tensor, ebits, mbits, qdim=-1)
        return dequant_tensor.to(fp32_tensor.dtype)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad, None, None
