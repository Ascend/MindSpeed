# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import logging
from typing import Optional, NamedTuple

import torch
import torch_npu

from megatron.training import get_args
from mindspeed.te.pytorch.module_typing import FP8Metadata
from mindspeed.te.pytorch.utils import get_hccl_comm_name, view_as_n_dim, all_gather_along_dim, get_quant_dtype

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
        fp8_scale = self.fp8_scale
        return Float8Tensor(
            data=data,
            fp8_dtype=self.fp8_dtype,
            fp8_scale=fp8_scale,
            dtype=self.dtype,
        )

    def quant_matmul(self, other: 'Float8Tensor', is_rowwise: tuple[bool, bool]):
        x1 = self.t() if is_rowwise[0] else self
        x2 = other.t() if is_rowwise[1] else other
        qdtype = get_quant_dtype()
        output = torch_npu.npu_quant_matmul(x1.data, x2.data, x2.fp8_scale,
                                            pertoken_scale=x1.fp8_scale,
                                            output_dtype=x1.dtype, **qdtype.mm_kwargs)
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


class QuantTensorMeta(NamedTuple):
    data: torch.Tensor
    scale: torch.Tensor

    def t(self):
        return self.data.T, self.scale.transpose(0, 1)


class Float8TensorWithTranspose:
    col_tensor: QuantTensorMeta
    row_tensor: QuantTensorMeta

    def __init__(
        self,
        fp8_dtype: torch.dtype,
        origin_shape: torch.Size,
        dtype: torch.dtype = torch.float32,
    ):
        self.fp8_dtype = fp8_dtype
        self.dtype = dtype
        self.origin_shape = origin_shape

    @property
    def device(self):
        return self.col_tensor.data.device

    def set_col_data(self, data, scale, t=False):
        if data is None:
            return
        self.col_tensor = QuantTensorMeta(
            data.T if t else data,
            scale.transpose(0, 1) if t else scale
        )

    def set_row_data(self, data, scale, t=False):
        if data is None:
            return
        self.row_tensor = QuantTensorMeta(
            data.T if t else data,
            scale.transpose(0, 1) if t else scale
        )

    def get_quant_data(self, is_rowwise=False):
        return self.row_tensor if is_rowwise else self.col_tensor

    def t(self):
        raise ValueError(f'{self.__class__.__name__} not support transpose')

    def quant_matmul(self, other: 'Float8TensorWithTranspose', is_rowwise):
        raise NotImplementedError()

    def restore_reshape(self, other: 'Float8TensorWithTranspose', output: torch.Tensor):
        if len(self.origin_shape) == len(other.origin_shape):
            return output  # dw 场景直接返回2维原始output
        return output.reshape(*self.origin_shape[:-1], *output.shape[1:])


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
