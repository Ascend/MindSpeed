# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import torch

from mindspeed.te.pytorch.fp8.tensor import MXFP8Tensor, Float8TensorWithTranspose


class MXFP8TensorCpu(Float8TensorWithTranspose):

    def from_MXFP8Tensor(self, mxfp8tensor: 'MXFP8Tensor'):
        self.fp8_dtype = mxfp8tensor.fp8_dtype
        self.data = mxfp8tensor.col_tensor.data.cpu()
        self.fp8_scale = mxfp8tensor.col_tensor.scale
        self.data_t = mxfp8tensor.row_tensor.data.cpu()
        self.scale_t = mxfp8tensor.row_tensor.scale
        self._dtype = mxfp8tensor.dtype
        self.origin_shape = mxfp8tensor.origin_shape

    def to_MXFP8Tensor(self):
        mxfp8tensor_npu = MXFP8Tensor(
            self.fp8_dtype,
            self.origin_shape,
            self._dtype
        )
        mxfp8tensor_npu.set_col_data(self.data, self.fp8_scale)
        mxfp8tensor_npu.set_row_data(self.data_t, self.scale_t)
        return mxfp8tensor_npu

    def padding_cpu_scale(self, mxfp8_tensor, scale_tensor):
        mxfp8_shape = mxfp8_tensor.shape
        scale_shape = scale_tensor.shape

        new_scale_shape = []
        padding_dim = -1
        for i, x in enumerate(mxfp8_shape):
            if x != scale_shape[i]:
                new_scale_shape.append(scale_shape[i] * scale_shape[-1])
                padding_dim = i
            else:
                new_scale_shape.append(scale_shape[i])

        scale_tensor = scale_tensor.view(new_scale_shape)

        scale_fp32 = scale_tensor.cpu().to(torch.float32)
        scale_fp32 = torch.repeat_interleave(scale_fp32, 32, dim=padding_dim)

        # Align shapes: crop or discard excess elements from x_scale_fp32 and weight_mxfp8_fp32
        if scale_fp32.shape[padding_dim] > mxfp8_tensor.shape[padding_dim]:
            scale_fp32 = scale_fp32.narrow(
                dim=padding_dim,
                start=0,
                length=mxfp8_tensor.shape[padding_dim]
            )

        return scale_fp32

    def quant_matmul(self, other, is_rowwise):
        x1, x_scale = self.get_quant_data(is_rowwise[0])
        x2, weight_scale = other.get_quant_data(is_rowwise[1])

        x_mxfp8_fp32 = x1.to(torch.float32).cpu()
        weight_mxfp8_fp32 = x2.to(torch.float32).cpu()

        x_scale_fp32 = self.padding_cpu_scale(x_mxfp8_fp32, x_scale)
        weight_scale_fp32 = self.padding_cpu_scale(weight_mxfp8_fp32, weight_scale)

        x_fp32 = torch.div(x_mxfp8_fp32, x_scale_fp32)
        weight_fp32 = torch.div(weight_mxfp8_fp32, weight_scale_fp32)
        output = torch.matmul(x_fp32, weight_fp32)
        return output
