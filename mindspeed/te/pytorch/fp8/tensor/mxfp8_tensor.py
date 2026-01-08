# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
from functools import partial
from typing import Tuple

import torch_npu
from mindspeed.args_utils import get_full_args as get_args
from mindspeed.te.pytorch.fp8.tensor.float8_tensor import Float8TensorWithTranspose
from mindspeed.te.pytorch.module_typing import FP8Metadata
from mindspeed.te.pytorch.utils import view_as_n_dim, get_hccl_comm_name, all_gather_along_dim


class MXFP8Tensor(Float8TensorWithTranspose):
    def t(self):
        raise ValueError('MXFP8 not support transpose')

    def quant_matmul(self, other: 'MXFP8Tensor', transpose=(False, False)):
        x1, x1_scale = self.get_by_trans(transpose[0])
        x2, x2_scale = other.get_by_trans(transpose[1])
        x1, x2 = map(view_as_n_dim, (x1, x2))
        x1_scale, x2_scale = map(partial(view_as_n_dim, dim=3), (x1_scale, x2_scale))
        output = torch_npu.npu_quant_matmul(x1, x2, x2_scale, pertoken_scale=x1_scale,
                                            output_dtype=self.dtype,
                                            scale_dtype=torch_npu.float8_e8m0fnu,
                                            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                                            group_sizes=[1, 1, 32])
        output = self.restore_reshape(output, transpose[0])
        # compare with cpu
        args = get_args()
        if args.te_comparison_with_cpu:
            from mindspeed.te.pytorch.fp8 import te_online_comparison_mxfp8_cpu
            te_online_comparison_mxfp8_cpu(self, other, transpose, output)
        if args.te_comparison_with_bf16:
            from mindspeed.te.pytorch.fp8 import te_online_comparison_mxfp8_bf16
            te_online_comparison_mxfp8_bf16(self, other, transpose, output)
        # 使用完 后续就不会继续调用, 直接清理掉显存
        for tensor in (x1, x1_scale, x2, x2_scale):
            tensor.untyped_storage().resize_(0)
        return output

    def all_gather_matmul(self, other: 'MXFP8Tensor', bias, fp8_meta: FP8Metadata, transpose: Tuple[bool, bool]):
        x1, x1_scale = self.get_by_trans()
        x2, x2_scale = other.get_by_trans(transpose[1])
        x1_t, x1_scale_t = self.get_by_trans(True)
        x1, x2, x1_t = map(view_as_n_dim, (x1, x2, x1_t))
        x1_scale, x2_scale, x1_scale_t = map(partial(view_as_n_dim, dim=3), (x1_scale, x2_scale, x1_scale_t))
        hcomm_name = get_hccl_comm_name(fp8_meta.tp_group, fp8_meta.tp_rank)
        if transpose == (False, False):  # 这里做兼容, 返还0轴做allgather
            x1_t = x1_t.t()
            x1_scale_t = x1_scale_t.transpose(0, 1)
        _, data_t = all_gather_along_dim(x1_t)
        _, scale_t = all_gather_along_dim(x1_scale_t)
        if transpose == (False, False):  # allgather之后仍然需要转置
            scale_t = scale_t.t()
            scale_t = scale_t.transpose(0, 1)
        output, _, _ = torch_npu.npu_all_gather_quant_mm(
            x1, x2,
            hcomm_name,
            fp8_meta.tp_world_size,
            bias=bias,
            x1_scale=x1_scale, x2_scale=x2_scale,
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
        gather_out = MXFP8Tensor(self.fp8_dtype, data_t=data_t, scale_t=scale_t, dtype=self.dtype)
        return output.view(-1, self.shape[1], output.shape[1]), gather_out

    def matmul_reduce_scatter(self, other: 'MXFP8Tensor', bias, fp8_meta: FP8Metadata, transpose: Tuple[bool, bool]):
        x1, x1_scale = self.get_by_trans(transpose[0])
        x2, x2_scale = other.get_by_trans(transpose[1])
        x1, x2 = map(view_as_n_dim, (x1, x2))
        x1_scale, x2_scale = map(partial(view_as_n_dim, dim=3), (x1_scale, x2_scale))
        hcomm_name = get_hccl_comm_name(fp8_meta.tp_group, fp8_meta.tp_rank)

        output, _ = torch_npu.npu_quant_mm_reduce_scatter(
            x1, x2, hcomm_name, fp8_meta.tp_world_size,
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
        return output.view(-1, self.shape[1], output.shape[1])
