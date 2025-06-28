# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Optional

import torch
import torch_npu



class Float8Tensor:

    def __init__(
        self,
        data: torch.Tensor,
        fp8_dtype: torch.dtype,
        fp8_scale_inv: Optional[torch.Tensor] = None,
        fp8_scale: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.data = data
        self.fp8_dtype = fp8_dtype
        self.fp8_scale_inv = fp8_scale_inv
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
        fp8_dtype: torch.dtype = None,
        scale: Optional[torch.Tensor] = None,
        scale_inv: Optional[torch.Tensor] = None,
    ):
        quant_tensor = tensor * scale
        quant_tensor = quant_tensor.to(dtype=fp8_dtype)
        return cls(
            data=quant_tensor,
            fp8_dtype=fp8_dtype,
            fp8_scale_inv=scale_inv,
            fp8_scale=scale,
            dtype=tensor.dtype,
        )

    def reshape(self, *args):
        self.data = self.data.reshape(*args)
        return self

    def view(self, *args):
        self.data = self.data.view(*args)
        return self

    def t(self):
        data = self.data.t()
        # 当前版本hif8的t()返回的是高精度版本
        if isinstance(self.data, torch_npu.HiFloat8Tensor):
            data = torch_npu.HiFloat8Tensor.to_hifloat8(data)
        if self.fp8_scale.numel() != 1:
            fp8_scale = self.fp8_scale.t()
            fp8_scale_inv = self.fp8_scale_inv.t()
        else:
            fp8_scale = self.fp8_scale
            fp8_scale_inv = self.fp8_scale_inv
        return Float8Tensor(
            data=data,
            fp8_dtype=self.fp8_dtype,
            fp8_scale_inv=fp8_scale_inv,
            fp8_scale=fp8_scale,
            dtype=self.dtype,
        )

    def quant_matmul(self, other):
        if isinstance(self.data, torch_npu.HiFloat8Tensor):
            output = torch_npu.npu_quant_matmul(self.data._data, other.data._data, other.fp8_scale,
                                                pertoken_scale=self.fp8_scale,
                                                output_dtype=self.dtype,
                                                x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8)
        else:
            output = torch_npu.npu_quant_matmul(self.data, other.data, other.fp8_scale,
                                            pertoken_scale=self.fp8_scale,
                                            output_dtype=self.dtype)
        return output
