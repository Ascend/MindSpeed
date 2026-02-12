# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
import torch_npu
from mindspeed.te.pytorch.utils import view_as_n_dim
from mindspeed.fsdp.quantization.utils import TensorWithTranspose


class MXTensor:
    def __init__(self, config):
        self.config = config
        self.fp8_format_dtype = None

    def to_mxfp8(self, data_hf, key):
        if data_hf is None:
            return data_hf
        ori_dtype = data_hf.dtype

        if data_hf.dtype == torch.float32:
            data_hf = data_hf.to(torch.bfloat16)

        self.fp8_format_dtype = self.config.get_key_dtype(key)
        if key == 'weight':
            y, mx_scale = torch_npu.npu_dynamic_mx_quant(data_hf, axis=-2, dst_type=self.fp8_format_dtype)

        else:
            y, mx_scale = torch_npu.npu_dynamic_mx_quant(data_hf, axis=-1, dst_type=self.fp8_format_dtype)

        if key == 'inputs':
            y_t, mx_scale_t = torch_npu.npu_dynamic_mx_quant(data_hf, axis=0, dst_type=self.fp8_format_dtype)

        elif key == 'grads':
            y_t, mx_scale_t = torch_npu.npu_dynamic_mx_quant(view_as_n_dim(data_hf), axis=-2,
                                                             dst_type=self.fp8_format_dtype)
            y_t, mx_scale_t = y_t.t(), mx_scale_t.transpose(0, 1)

        else:
            y_t, mx_scale_t = torch_npu.npu_dynamic_mx_quant(view_as_n_dim(data_hf), axis=-1,
                                                             dst_type=self.fp8_format_dtype)
            y_t, mx_scale_t = y_t.t(), mx_scale_t.transpose(0, 1)

        return TensorWithTranspose(self.fp8_format_dtype, y, mx_scale, y_t, mx_scale_t, ori_dtype)