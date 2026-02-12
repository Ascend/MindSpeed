# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import torch_npu
from mindspeed.args_utils import get_full_args as get_args
from mindspeed.te.pytorch.fp8.tensor.float8_tensor import Float8Tensor2D
from mindspeed.te.pytorch.module_typing import FP8Metadata
from mindspeed.te.pytorch.utils import all_gather_along_dim


class MXFP8Tensor(Float8Tensor2D):

    def quant_matmul(self, other: 'MXFP8Tensor', is_rowwise):
        x1, x1_scale = self.get_quant_data(is_rowwise[0])
        x2, x2_scale = other.get_quant_data(is_rowwise[1])
        output = torch_npu.npu_quant_matmul(x1, x2, x2_scale, pertoken_scale=x1_scale,
                                            output_dtype=self.dtype,
                                            scale_dtype=torch_npu.float8_e8m0fnu,
                                            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                                            group_sizes=[1, 1, 32])
        output = self.restore_reshape(other, output)
        # compare with cpu
        args = get_args()
        if args.te_comparison_with_cpu:
            from mindspeed.te.pytorch.fp8 import te_online_comparison_mxfp8_cpu
            te_online_comparison_mxfp8_cpu(self, other, is_rowwise, output)
        if args.te_comparison_with_bf16:
            from mindspeed.te.pytorch.fp8 import te_online_comparison_mxfp8_bf16
            te_online_comparison_mxfp8_bf16(self, other, is_rowwise, output)
        # 使用完 后续就不会继续调用, 直接清理掉显存
        for tensor in (x1, x1_scale, x2, x2_scale):
            tensor.untyped_storage().resize_(0)
        return output

    def all_gather_matmul(self, other: 'MXFP8Tensor', bias, fp8_meta: FP8Metadata, transpose: tuple[bool, bool]):
        x2, x2_scale = other.get_quant_data(transpose[1])
        row_data, row_scale = self.row_tensor.t() if transpose == (False, False) else self.row_tensor
        _, row_data = all_gather_along_dim(row_data)
        _, row_scale = all_gather_along_dim(row_scale)
        output, _, _ = torch_npu.npu_all_gather_quant_mm(
            self.col_tensor.data, x2,
            fp8_meta.hcom_name,
            fp8_meta.tp_world_size,
            bias=bias,
            x1_scale=self.col_tensor.scale,
            x2_scale=x2_scale,
            quant_scale=None,
            block_size=0,
            comm_turn=0,
            group_sizes=[1, 1, 32],
            amax_output=False,
            y_dtype=self.dtype,
            gather_output=False,
            x1_dtype=None, x2_dtype=None,
            x1_scale_dtype=torch_npu.float8_e8m0fnu,
            x2_scale_dtype=torch_npu.float8_e8m0fnu,
        )
        gather_out = MXFP8Tensor(self.fp8_dtype, self.origin_shape, dtype=self.dtype)
        gather_out.set_row_data(row_data, row_scale, transpose == (False, False))
        return output.view(-1, self.origin_shape[1], output.shape[1]), gather_out

    def matmul_reduce_scatter(self, other: 'MXFP8Tensor', bias, fp8_meta: FP8Metadata, transpose: tuple[bool, bool]):
        x1, x1_scale = self.get_quant_data(transpose[0])
        x2, x2_scale = other.get_quant_data(transpose[1])

        output, _ = torch_npu.npu_quant_mm_reduce_scatter(
            x1, x2,
            fp8_meta.hcom_name,
            fp8_meta.tp_world_size,
            bias=bias,
            reduce_op='sum',
            x1_scale=x1_scale, x2_scale=x2_scale,
            quant_scale=None,
            block_size=0,
            comm_turn=0,
            group_sizes=[1, 1, 32],
            amax_output=False,
            y_dtype=self.dtype,
            x1_dtype=None, x2_dtype=None,
            x1_scale_dtype=torch_npu.float8_e8m0fnu,
            x2_scale_dtype=torch_npu.float8_e8m0fnu,
        )
        return output.view(-1, self.origin_shape[1], output.shape[1])
