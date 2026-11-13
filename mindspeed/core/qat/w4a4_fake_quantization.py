# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
