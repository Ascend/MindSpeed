# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Optional

import torch

from mindspeed.te.pytorch.fp8.constants import FP8Format
from mindspeed.te.pytorch.fp8.tensor import Float8Tensor


class Float8TensorCpu:

    def __init__(
        self,
        data: torch.Tensor,
        fp8_dtype: torch.dtype,
        fp8_scale: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.data = data
        self.fp8_dtype = fp8_dtype
        self.fp8_scale = fp8_scale
        self._dtype = dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self._dtype

    @classmethod
    def to_float8(
        cls,
        tensor: torch.Tensor,
        *,
        fp8_format: FP8Format = None,
        scale: Optional[torch.Tensor] = None,
        dtype=None
    ):
        if dtype is None:
            dtype = tensor.dtype
        from mindspeed.te.pytorch.fp8 import cast_to_fp8_cpu
        tensor_cpu = tensor.cpu()
        quant_tensor = cast_to_fp8_cpu(tensor_cpu, fp8_format)
        return cls(
            data=quant_tensor,
            fp8_dtype=fp8_format.dtype,
            fp8_scale=scale,
            dtype=dtype,
        )

    def from_float8tensor(self, float8tensor: 'Float8Tensor'):
        self.data = float8tensor.data.cpu()
        self.fp8_dtype = float8tensor.fp8_dtype
        self.fp8_scale = float8tensor.fp8_scale
        self._dtype = float8tensor.dtype

    def to_float8tensor(self):
        float8tensor_npu = Float8Tensor(
            self.data.npu(),
            self.fp8_dtype,
            self.fp8_scale,
            self._dtype
        )
        return float8tensor_npu

    def reshape(self, *args):
        self.data = self.data.reshape(*args)
        return self

    def view(self, *args):
        return self.data.view(*args)

    def t(self):
        data = self.data.t()
        # CPU版本直接返回转置后的FP8数据
        if self.fp8_scale.numel() != 1:
            fp8_scale = self.fp8_scale.t()
        else:
            fp8_scale = self.fp8_scale
        return Float8Tensor(
            data=data,
            fp8_dtype=self.fp8_dtype,
            fp8_scale=fp8_scale,
            dtype=self.dtype,
        )

    def quant_matmul(self, other):
        # CPU上FP8 matmul
        self.data = self.data.cpu()
        other.data = other.data.cpu()
        quantized_result = torch.matmul(self.data, other.data)

        # 反量化
        dequantized_result = self.cpu_fp8_dequantize(quantized_result)
        return dequantized_result.to(dtype=torch.bfloat16)

    def cpu_fp8_dequantize(self, quantized_tensor):
        # FP8反量化实现
        if self.fp8_scale.numel() != 1:
            raise ValueError("not yet supported")
        if self.fp8_scale.item() == 0:
            return torch.zeros_like(quantized_tensor, dtype=self.dtype)

        quantized_tensor = quantized_tensor.cpu()
        dequant_tensor = quantized_tensor / self.fp8_scale
        return dequant_tensor.to(dtype=self.dtype)