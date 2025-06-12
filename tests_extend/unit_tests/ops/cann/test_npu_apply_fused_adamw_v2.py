# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch_npu
import pytest
from mindspeed.ops.npu_apply_fused_adamw_v2 import npu_apply_fused_adamw_v2


@dataclass
class AdamWParameters:
    var: any = 0.5
    m: any = 0.9
    v: any = 0.9
    max_grad_norm: any = 0.9
    grad: any = 0.5
    step: any = 10
    lr: float = 1e-3
    beta1: float = 0.9999
    beta2: float = 0.9999
    weight_decay: float = 0.0
    eps: float = 1e-8
    amsgrad: bool = False
    maximize: bool = False
    shape: int = 10


def single_tensor_adamw(*args):
    (param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, step_t,
     lr, beta1, beta2, weight_decay, eps, amsgrad, maximize) = args
    dtype1 = param.dtype
    dtype2 = grad.dtype

    lr = np.float32(lr)
    beta1 = np.float32(beta1)
    beta2 = np.float32(beta2)
    weight_decay = np.float32(weight_decay)
    eps = np.float32(eps)

    if dtype1 != dtype2:
        grad = grad.to(dtype1)
        max_exp_avg_sq = max_exp_avg_sq.to(dtype1)
    if maximize:
        grad = -grad

    step = step_t
    step = step.item()

    param = param * (1 - lr * weight_decay)

    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    step_size = lr / bias_correction1
    bias_correction2_sqrt = math.sqrt(bias_correction2)
    if amsgrad:
        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps
    else:
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps

    param.addcdiv_(exp_avg, denom, value=-step_size)

    if dtype1 != dtype2:
        max_exp_avg_sq = max_exp_avg_sq.to(dtype2)
    return param, exp_avg, exp_avg_sq, max_exp_avg_sq


def cpu_apply_fused_adamw_v2(*args):
    (var, grad, m, v, max_grad_norm, step,
     lr, beta1, beta2, weight_decay, eps, amsgrad, maximize) = args
    var_dtype, m_dtype, v_dtype, grad_dtype, step_dtype, max_grad_norm_dtype = \
        var.dtype, m.dtype, v.dtype, grad.dtype, step.dtype, max_grad_norm.dtype
    # perform high precision conversion
    is_var_dtype_bf16_fp16 = "bfloat16" in str(var_dtype) or "float16" in str(var_dtype)
    is_grad_dtype_bf16_fp16 = "bfloat16" in str(grad_dtype) or "float16" in str(grad_dtype)
    if is_var_dtype_bf16_fp16:
        adamw_params = [
            var.to(torch.float32), grad.to(torch.float32), m.to(torch.float32), v.to(torch.float32),
            max_grad_norm.to(torch.float32), step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize
        ]
    elif is_grad_dtype_bf16_fp16:
        adamw_params = [
            var, grad.to(torch.float32), m, v, max_grad_norm.to(torch.float32), step, lr, beta1, beta2,
            weight_decay, eps, amsgrad, maximize
        ]
    else:
        adamw_params = [
            var, grad, m, v, max_grad_norm, step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize
        ]
    if "int64" in str(step_dtype):
        step_fp32 = step.to(torch.float32)
        adamw_params[5] = step_fp32
    # perform adamw
    res_var, res_m, res_v, res_max_grad_norm = single_tensor_adamw(*adamw_params)
    # perform original precision conversion
    if is_var_dtype_bf16_fp16:
        res_var, res_m, res_v, res_max_grad_norm = (
            res_var.to(var_dtype), res_m.to(var_dtype),
            res_v.to(var_dtype), res_max_grad_norm.to(max_grad_norm_dtype)
        )
    elif is_grad_dtype_bf16_fp16:
        res_max_grad_norm = res_max_grad_norm.to(max_grad_norm_dtype)
    # copy result back to input reference
    var.copy_(res_var)
    m.copy_(res_m)
    v.copy_(res_v)
    max_grad_norm.copy_(res_max_grad_norm)


def generate_data(p, dtype_list):
    # generate npu/cpu test data
    cpu_data, npu_data, shape = [], [], p.shape
    for attr, dtype in zip(['var', 'grad', 'm', 'v', 'max_grad_norm'], dtype_list):
        cpu_data.append(torch.full((shape, shape), getattr(p, attr)).to(dtype))
        npu_data.append(torch.full((shape, shape), getattr(p, attr)).to(dtype).to('npu'))
    cpu_data.append(torch.full((1,), p.step).to(torch.int64))
    npu_data.append(torch.full((1,), p.step).to(torch.int64).to('npu'))
    cpu_data.extend([p.lr, p.beta1, p.beta2, p.weight_decay, p.eps, p.amsgrad, p.maximize])
    npu_data.extend([p.lr, p.beta1, p.beta2, p.weight_decay, p.eps, p.amsgrad, p.maximize])
    return cpu_data, npu_data


def output_check(output):
    #  using different criterion according to tensor dtype
    total = 0
    criterion = {'torch.float32': 2 ** (-11), 'torch.float16': 2 ** (-9), 'torch.bfloat16': 2 ** (-8) + 2 ** (-14)}
    for npu_out, cpu_out in output:
        crit = criterion.get(str(npu_out.dtype))
        count = (torch.sum(torch.isnan(npu_out)).item(),
                 torch.sum(torch.isinf(npu_out)).item(),
                 torch.sum(torch.abs(npu_out - cpu_out.npu()) > crit).item())
        total += sum(count)
    return total


def run_ops(param_data_type, grad_data_type, max_grad_norm_data_type):
    params = AdamWParameters()
    dtype_list = [param_data_type, grad_data_type, param_data_type, param_data_type, max_grad_norm_data_type]
    cpu_data, npu_data = generate_data(params, dtype_list)
    npu_apply_fused_adamw_v2(*npu_data)
    cpu_apply_fused_adamw_v2(*cpu_data)
    output = list(zip(npu_data, cpu_data))[:5]
    if output_check(output) != 0:
        raise AssertionError('NPU and CPU results ars inconsistent.')


class TestNpuApplyFusedAdamWV2:
    def test_ops(self):
        case_list = []
        for param_data_type in (torch.float32, torch.float16, torch.bfloat16):
            for grad_data_type in (torch.float32, torch.float16, torch.bfloat16):
                for max_grad_norm_data_type in (torch.float32, torch.float16, torch.bfloat16):
                    case_list.append([param_data_type, grad_data_type, max_grad_norm_data_type])
        for case in case_list:
            run_ops(*case)
