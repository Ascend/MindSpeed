# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
from functools import partial

import torch_npu

from mindspeed.te.pytorch.fp8.tensor.float8_tensor import Float8TensorWithTranspose
from mindspeed.te.pytorch.utils import view_as_n_dim


class MXFP8Tensor(Float8TensorWithTranspose):
    def t(self):
        raise ValueError('MXFP8 not support transpose')

    def quant_matmul(self, other: Float8TensorWithTranspose, transpose=(False, False)):
        x1, x1_scale = self.get_by_trans(transpose[0])
        x2, x2_scale = other.get_by_trans(transpose[1])
        x1, x2 = map(view_as_n_dim, (x1, x2))
        x1_scale, x2_scale = map(partial(view_as_n_dim, dim=3), (x1_scale, x2_scale))
        output = torch_npu.npu_quant_matmul(x1, x2, x2_scale, pertoken_scale=x1_scale,
                                            output_dtype=self.dtype,
                                            scale_dtype=torch_npu.float8_e8m0fnu,
                                            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                                            group_sizes=[1, 1, 32])
        return self.restore_reshape(output, transpose[0])
