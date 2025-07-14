# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch

__all__ = ["npu_apply_fused_adamw_v2"]

from mindspore.ops import auto_generate as gen
adamw_opt = gen.AdamW()


def npu_apply_fused_adamw_v2(param: torch.Tensor,
                             grad: torch.Tensor,
                             exp_avg: torch.Tensor,
                             exp_avg_sq: torch.Tensor,
                             max_exp_avg_sq: torch.Tensor,
                             state_step: torch.Tensor,
                             lr: float = 1e-3,
                             beta1: float = 0.9,
                             beta2: float = 0.999,
                             weight_decay: float = 0.0,
                             eps: float = 1e-8,
                             amsgrad: bool = False,
                             maximize: bool = False,
                             ):
    success = True
    if max_exp_avg_sq is None:
        adamw_opt(param, exp_avg, exp_avg_sq, exp_avg_sq, grad, state_step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)
    else:
        adamw_opt(param, exp_avg, exp_avg_sq, max_exp_avg_sq, grad, state_step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)
    return success