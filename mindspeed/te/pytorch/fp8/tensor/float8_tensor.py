# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import logging
from typing import Optional

import torch
import torch_npu

from megatron.training import get_args
from mindspeed.te.pytorch.fp8.constants import FP8Format
from mindspeed.te.pytorch.module_typing import FP8Metadata
from mindspeed.te.pytorch.utils import get_hccl_comm_name, view_as_n_dim, all_gather_along_dim

logger = logging.getLogger(__name__)


class Float8Tensor:

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
        from mindspeed.te.pytorch.fp8 import cast_to_fp8
        quant_tensor = cast_to_fp8(tensor, fp8_format)
        if get_args().te_comparison_with_cpu:
            te_cast_comparison(fp8_format, tensor, quant_tensor)
        return cls(
            data=quant_tensor,
            fp8_dtype=fp8_format.dtype,
            fp8_scale=scale,
            dtype=dtype,
        )

    def reshape(self, *args):
        self.data = self.data.reshape(*args)
        return self

    def view(self, *args):
        return self.__class__(
            data=self.data.view(*args),
            fp8_dtype=self.fp8_dtype,
            fp8_scale=self.fp8_scale,
            dtype=self.dtype,
        )

    def t(self):
        data = self.data.t()
        # 当前版本hif8的t()返回的是高精度版本
        if isinstance(self.data, torch_npu.HiFloat8Tensor):
            data = torch_npu.HiFloat8Tensor.to_hifloat8(data)
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

    def quant_matmul(self, other, transpose):
        x1 = self.t() if transpose[0] else self
        x2 = other.t() if transpose[1] else other
        if isinstance(x1.data, torch_npu.HiFloat8Tensor):
            output = torch_npu.npu_quant_matmul(x1.data._data, x2.data._data, x2.fp8_scale,
                                                pertoken_scale=x1.fp8_scale,
                                                output_dtype=x1.dtype,
                                                x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8)
        # blockwise tensor 3d to 2d
        elif x1.fp8_scale.numel() != 1:
            data = x1.data.view([-1, x1.data.shape[-1]])
            fp8_scale = x1.fp8_scale.contiguous()
            output = torch_npu.npu_quant_matmul(data, x2.data, x2.fp8_scale,
                                                pertoken_scale=fp8_scale,
                                                output_dtype=x1.dtype,
                                                group_sizes=[128, 128, 128])
            output = output.view([*x1.data.shape[:-1], output.shape[-1]])
        else:
            output = torch_npu.npu_quant_matmul(x1.data, x2.data, x2.fp8_scale,
                                                pertoken_scale=x1.fp8_scale,
                                                output_dtype=x1.dtype)
        # te cpu compare
        args = get_args()
        if args.te_comparison_with_cpu:
            from mindspeed.te.pytorch.fp8 import te_online_comparison_cpu
            te_online_comparison_cpu(x1, x2, output)
        if args.te_comparison_with_bf16:
            from mindspeed.te.pytorch.fp8 import te_online_comparison_bf16
            te_online_comparison_bf16(x1, x2, output)
        return output

    def all_gather_matmul(self, other: 'Float8Tensor', bias, fp8_meta: FP8Metadata, transpose):
        x1, x1_scale = map(view_as_n_dim, (self.data, self.fp8_scale))
        _, x1_scale = all_gather_along_dim(x1_scale)
        x2, x2_scale = map(view_as_n_dim, (other.data, other.fp8_scale))
        # x1 scale 因为是单标量tensor 与之前allgather输出 tensor相同 可以省一次allgather
        hcomm_name = get_hccl_comm_name(fp8_meta.tp_group, fp8_meta.tp_rank)
        output, gather_out, _ = torch_npu.npu_all_gather_quant_mm(
            x1, x2, hcomm_name, fp8_meta.tp_world_size,
            bias=bias, x1_scale=x1_scale, x2_scale=x2_scale,
            y_dtype=self.dtype
        )
        gather_out = Float8Tensor(gather_out.t(), self.fp8_dtype, x1_scale, self.dtype)
        return output.view(-1, self.shape[1], output.shape[1]), gather_out

    def matmul_reduce_scatter(self, other: 'Float8Tensor', bias, fp8_meta: FP8Metadata, transpose=(False, False)):
        x1, x1_scale = map(view_as_n_dim, (self.data, self.fp8_scale))
        x2, x2_scale = map(view_as_n_dim, (other.data, other.fp8_scale))
        if transpose[0]:
            x1, x1_scale = x1.T, x1_scale.T
        if transpose[1]:
            x2, x2_scale = x2.T, x2_scale.T

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
        )
        return output.view(-1, self.shape[1], output.shape[1])


class Float8TensorWithTranspose(Float8Tensor):
    def __init__(
        self,
        fp8_dtype: torch.dtype,
        data: torch.Tensor = None,
        scale: Optional[torch.Tensor] = None,
        data_t: torch.Tensor = None,
        scale_t: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super(Float8TensorWithTranspose, self).__init__(data, fp8_dtype, scale, dtype)
        self.data_t = data_t
        self.scale_t = scale_t

    def t(self):
        raise NotImplementedError()

    def quant_matmul(self, other: 'Float8Tensor', transpose=(False, False)):
        raise NotImplementedError()

    def get_by_trans(self, transpose=False):
        if transpose:
            return self.data_t, self.scale_t
        return self.data, self.fp8_scale

    def restore_reshape(self, output, transpose=False):
        x, _ = self.get_by_trans(transpose)
        if len(x.shape) == 2:
            return output
        return output.reshape(*x.shape[:-1], *output.shape[1:])


def te_cast_comparison(fp8_format, tensor, quant_tensor):
    from mindspeed.te.pytorch.fp8 import cast_to_fp8_cpu
    if fp8_format.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(
            f"TE online comparison only supports e4m3 and e5m2 formats, but fp8_dtype is {fp8_format.dtype}")
    tensor_cpu = tensor.cpu()
    quant_tensor_cpu = cast_to_fp8_cpu(tensor_cpu, fp8_format)

    quant_tensor_cpu = quant_tensor_cpu.npu()
    abs_error = torch.abs(quant_tensor_cpu.to(torch.float32) - quant_tensor.to(torch.float32))
    rel_error = abs_error / torch.abs(quant_tensor_cpu.to(torch.float32))
    max_abs_error = torch.max(abs_error)
    max_rel_error = torch.max(rel_error)

    logger.info("The error of cast to fp8: ")
    logger.info(f"[{quant_tensor.device}] Max Absolute Error: {max_abs_error.item()}")
    logger.info(f"[{quant_tensor.device}] Max Relative Error: {max_rel_error.item()}")
    if max_rel_error > 0.0:
        raise ValueError(f"The error of cast exceeds tolerance: {max_rel_error.item()}")
