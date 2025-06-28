# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Optional

import torch
import torch_npu


class MXFP8Tensor:
    def __init__(
            self,
            fp8_dtype: torch.dtype,
            data: torch.Tensor,
            scale: Optional[torch.Tensor],
            data_t: torch.Tensor,
            scale_t: Optional[torch.Tensor],
            dtype: torch.dtype = torch.float32,
    ):
        self.fp8_dtype = fp8_dtype
        self.data = data
        self.scale = scale
        self.data_t = data_t
        self.scale_t = scale_t
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

    def reshape(self, *args):
        self.data = self.data.reshape(*args)
        return self

    def view(self, *args):
        self.data = self.data.view(*args)
        return self

    def t(self):
        raise ValueError('MXFP8 not support transpose')

    def quant_matmul(self, other, transpose=(False, False)):
        x1, x1_scale = (self.data_t, self.scale_t) if transpose[0] else (self.data, self.scale)
        # 当前quantmatmul算子仅支持MK*NK，因此x2的转置取反
        x2, x2_scale = (other.data_t, other.scale_t) if transpose[1] else (other.data, other.scale)
        if len(x2.shape) != 2:
            x2, x2_scale = reshape_to_2D(x2), reshape_to_2D(x2_scale)

        output = torch_npu.npu_quant_matmul(x1, x2, x2_scale, pertoken_scale=x1_scale,
                                            output_dtype=self.dtype,
                                            scale_dtype=torch_npu.float8_e8m0,
                                            pertoken_scale_dtype=torch_npu.float8_e8m0)
        return output


def reshape_to_2D(input_tensor):
    # Convert the tensor shapes to 2D for execution compatibility
    if len(input_tensor.shape) != 2:
        input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])
    return input_tensor
