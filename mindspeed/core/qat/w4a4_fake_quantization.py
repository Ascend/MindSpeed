# coding=utf-8
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""
W4A4 Fake Quantization for Activation.

Wraps the MXFP4 E2M1 quantization kernel in a separate Autograd Function for:
1. Clear separation of concerns (activation vs weight quantization)
2. Independent control of STE behavior
3. Future extensibility (e.g., Hadamard transform, stochastic rounding)
"""

from torch.autograd import Function
from mindspeed.core.qat.w4a16_fake_quantization import w4a16_fake_quant


class W4A4FakeQuantization(Function):
    """Activation fake quantization with STE backward.

    Forward:  quantize activation to MXFP4 E2M1 format
    Backward: STE — gradient passes through unchanged
    """

    @staticmethod
    def forward(ctx, fp32_tensor, block_size, transpose):
        ebits, mbits = 2.0, 3.0  # MXFP4 E2M1
        dequant_tensor = w4a16_fake_quant(fp32_tensor, ebits, mbits, qdim=-1)
        return dequant_tensor.to(fp32_tensor.dtype)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad, None, None
