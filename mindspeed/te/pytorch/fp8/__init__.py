import logging
import math

import numpy as np
import torch

import torch_npu
from mindspeed.args_utils import get_full_args as get_args
from mindspeed.te.pytorch.fp8.constants import FormatEnum
from mindspeed.te.pytorch.fp8.tensor import Float8Tensor, Float8TensorCpu, MXFP8Tensor, MXFP8TensorCpu, Float8TensorWithTranspose

logger = logging.getLogger(__name__)


def fp8_matmul(inputs, weight, fp8_meta, key, transpose=(False, False)):
    from mindspeed.te.pytorch.fp8.recipes import MXFP8BlockScaling, GroupwiseBlockScaling

    if isinstance(fp8_meta.fp8_recipe, MXFP8BlockScaling) or isinstance(fp8_meta.fp8_recipe, GroupwiseBlockScaling):
        if not isinstance(inputs, Float8TensorWithTranspose):
            inputs = fp8_meta.pre_compute(key[0], inputs)
        if not isinstance(weight, Float8TensorWithTranspose):
            weight = fp8_meta.pre_compute(key[1], weight)

        # quant matmul with transpose
        if key == ('grads', 'inputs'):
            # EVB 调测算子能力不支持transpose临时规避
            transpose = (True, True)
        output = inputs.quant_matmul(weight, transpose)

        args = get_args()
        if args.te_comparison_with_cpu:
            te_online_comparison_mxfp8_cpu(inputs, weight, transpose, output)
        if args.te_comparison_with_bf16:
            te_online_comparison_mxfp8_bf16(inputs, weight, transpose, output)

    else:
        if not isinstance(inputs, Float8Tensor):
            inputs = fp8_meta.pre_compute(key[0], inputs)
        if not isinstance(weight, Float8Tensor):
            weight = fp8_meta.pre_compute(key[1], weight)
        inputs = inputs.t() if transpose[0] else inputs
        weight = weight.t() if transpose[1] else weight

        # quant matmul
        output = inputs.quant_matmul(weight)

        args = get_args()
        if args.te_comparison_with_cpu:
            te_online_comparison_cpu(inputs, weight, output)
        if args.te_comparison_with_bf16:
            te_online_comparison_bf16(inputs, weight, output)
    return output


def te_online_comparison_cpu(inputs, weight, output):
    if inputs.fp8_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(
            f"TE online comparison only supports e4m3 and e5m2 formats, "
            f"but fp8_dtype is {inputs.fp8_dtype}")

    # convert Float8Tensor to Float8TensorCPU
    weight_cpu = Float8TensorCpu(
        torch.tensor([]),
        torch.float8_e4m3fn,
        None,
        torch.float32
    )
    input_cpu = Float8TensorCpu(
        torch.tensor([]),
        torch.float8_e4m3fn,
        None,
        torch.float32
    )
    weight_cpu.from_float8tensor(weight)
    input_cpu.from_float8tensor(inputs)

    output_cpu = input_cpu.quant_matmul(weight_cpu).npu()

    abs_error = torch.abs(output_cpu - output)
    rel_error = abs_error / torch.abs(output_cpu)
    max_abs_error = torch.max(abs_error)
    max_rel_error = torch.max(rel_error)

    logger.info(f"The error of quant_matmul: ")
    logger.info(f"[{output.device}] Max Absolute Error: {max_abs_error.item()}")
    logger.info(f"[{output.device}] Max Relative Error: {max_rel_error.item()}")
    if max_rel_error > 0.001:
        raise ValueError(f"The error of quant_matmul exceeds tolerance: {max_rel_error.item()}")


def te_online_comparison_bf16(inputs, weight, output):
    inputs_bf16 = inputs.data.to(torch.float32).to(torch.bfloat16)
    weight_bf16 = weight.data.to(torch.float32).to(torch.bfloat16)
    bf16_output = torch.matmul(inputs_bf16, weight_bf16)

    abs_error = torch.abs(bf16_output - output)
    rel_error = abs_error / torch.abs(bf16_output)
    max_abs_error = torch.max(abs_error)
    max_rel_error = torch.max(rel_error)
    logger.info(f"The error of quant_matmul: ")
    logger.info(f"[{output.device}] Max Absolute Error: {max_abs_error.item()}")
    logger.info(f"[{output.device}] Max Relative Error: {max_rel_error.item()}")
    if max_rel_error > 0.001:
        raise ValueError(f"The error of quant_matmul exceeds tolerance: {max_rel_error.item()}")


def te_online_comparison_mxfp8_cpu(inputs, weight, transpose, output):
    if inputs.fp8_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(
            f"TE online comparison only supports e4m3 and e5m2 formats, "
            f"but fp8_dtype is {inputs.fp8_dtype}")

    # convert Float8Tensor to MXFP8TensorCpu
    weight_cpu = MXFP8TensorCpu(
        torch.float8_e4m3fn,
        torch.tensor([]),
        None,
        torch.tensor([]),
        None,
        torch.float32
    )
    input_cpu = MXFP8TensorCpu(
        torch.float8_e4m3fn,
        torch.tensor([]),
        None,
        torch.tensor([]),
        None,
        torch.float32
    )
    weight_cpu.from_MXFP8Tensor(weight)
    input_cpu.from_MXFP8Tensor(inputs)

    output_cpu = input_cpu.quant_matmul(weight_cpu, transpose).npu()

    abs_error = torch.abs(output_cpu - output)
    rel_error = abs_error / torch.abs(output_cpu)
    max_abs_error = torch.max(abs_error)
    max_rel_error = torch.max(rel_error)

    logger.info(f"The error of quant_matmul: ")
    logger.info(f"[{output.device}] Max Absolute Error: {max_abs_error.item()}")
    logger.info(f"[{output.device}] Max Relative Error: {max_rel_error.item()}")
    if max_rel_error > 0.001:
        raise ValueError(f"The error of quant_matmul exceeds tolerance: {max_rel_error.item()}")


def padding_bf16_scale(mxfp8_tensor, scale_tensor):
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

    scale_bf16 = scale_tensor.to(torch.float32).to(torch.bfloat16)
    scale_bf16 = torch.repeat_interleave(scale_bf16, 32, dim=padding_dim)

    # Align shapes: crop or discard excess elements from x_scale_bf16 and weight_mxfp8_bf16
    if scale_bf16.shape[padding_dim] > mxfp8_tensor.shape[padding_dim]:
        scale_bf16 = scale_bf16.narrow(
            dim=padding_dim,
            start=0,
            length=mxfp8_tensor.shape[padding_dim]
        )

    return scale_bf16


def te_online_comparison_mxfp8_bf16(inputs, weight, transpose, output):
    x1, x_scale = inputs.get_by_trans(transpose[0])
    x2, weight_scale = weight.get_by_trans(transpose[1])

    x_mxfp8_bf16 = x1.to(torch.float32).to(torch.bfloat16)
    weight_mxfp8_bf16 = x2.to(torch.float32).to(torch.bfloat16)

    x_scale_bf16 = padding_bf16_scale(x_mxfp8_bf16, x_scale)
    weight_scale_bf16 = padding_bf16_scale(weight_mxfp8_bf16, weight_scale)

    x_bf16 = torch.div(x_mxfp8_bf16, x_scale_bf16)
    weight_bf16 = torch.div(weight_mxfp8_bf16, weight_scale_bf16)
    bf16_output = torch.matmul(x_bf16, weight_bf16)

    abs_error = torch.abs(bf16_output - output)
    rel_error = abs_error / torch.abs(bf16_output)
    max_abs_error = torch.max(abs_error)
    max_rel_error = torch.max(rel_error)
    logger.info(f"The error of quant_matmul: ")
    logger.info(f"[{output.device}] Max Absolute Error: {max_abs_error.item()}")
    logger.info(f"[{output.device}] Max Relative Error: {max_rel_error.item()}")
    if max_rel_error > 0.001:
        raise ValueError(f"The error of quant_matmul exceeds tolerance: {max_rel_error.item()}")


class Cast2FP8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quant_tensor, fp8_format):
        if fp8_format == FormatEnum.E4M3.value:
            return quant_tensor.to(torch.float8_e4m3fn)
        elif fp8_format == FormatEnum.E5M2.value:
            return quant_tensor.to(torch.float8_e5m2)
        elif fp8_format == FormatEnum.HIF8.value:
            return torch_npu.HiFloat8Tensor.to_hifloat8(quant_tensor)
        else:
            raise ValueError("Only e4m3, e5m2 and hif8 of the fp8 format are supported.")

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def cast_to_fp8(quant_tensor, fp8_format):
    return Cast2FP8.apply(quant_tensor, fp8_format)


class Cast2FP8Cpu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quant_tensor, fp8_format):
        if fp8_format == FormatEnum.E4M3.value:
            quantized_tensor = quant_tensor.to(torch.float8_e4m3fn)
        elif fp8_format == FormatEnum.E5M2.value:
            quantized_tensor = quant_tensor.to(torch.float8_e5m2)
        else:
            raise ValueError("Only e4m3, e5m2 of the fp8 format are supported on cpu.")

        return quantized_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def cast_to_fp8_cpu(quant_tensor, fp8_format):
    return Cast2FP8Cpu.apply(quant_tensor, fp8_format)
