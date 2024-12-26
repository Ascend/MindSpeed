# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import warnings
from dataclasses import dataclass
import torch
import torch_npu
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)
from mindspeed.ops.npu_apply_fused_ema_adamw import npu_apply_fused_ema_adamw


@dataclass
class Params:
    grad: float = 0.5
    var: float = 0.5
    m: float = 0.9
    v: float = 0.9
    s: float = 0.9
    step: int = 10
    lr: float = 1e-3
    ema_decay: float = 0.9999
    beta1: float = 0.9999
    beta2: float = 0.9999
    eps: float = 1e-8
    mode: int = 1
    bias_correction: bool = True
    weight_decay: float = 0.0
    shape: int = 10


class TestNpuApplyFusedEmaAdamW:
    crit = [2 ** (-14), 2 ** (-13), 2 ** (-12)]

    @pytest.mark.skip(reason='temporary skip for compatibility with released CANN.')
    def test_ops(self):
        p = Params
        cpu_grad = torch.full((p.shape, p.shape), p.grad).to(torch.float32)
        cpu_var = torch.full((p.shape, p.shape), p.var).to(torch.float32)
        cpu_m = torch.full((p.shape, p.shape), p.m).to(torch.float32)
        cpu_v = torch.full((p.shape, p.shape), p.v).to(torch.float32)
        cpu_s = torch.full((p.shape, p.shape), p.s).to(torch.float32)
        cpu_step = torch.full((1,), p.step).to(torch.int64)
        npu_grad = torch.full((p.shape, p.shape), p.grad).to(torch.float32).npu()
        npu_var = torch.full((p.shape, p.shape), p.var).to(torch.float32).npu()
        npu_m = torch.full((p.shape, p.shape), p.m).to(torch.float32).npu()
        npu_v = torch.full((p.shape, p.shape), p.v).to(torch.float32).npu()
        npu_s = torch.full((p.shape, p.shape), p.s).to(torch.float32).npu()
        npu_step = torch.full((1,), p.step).to(torch.int64).npu()
        cpu_var, cpu_m, cpu_v, cpu_s = TestNpuApplyFusedEmaAdamW.cpu_ema_adamw(
            cpu_grad, cpu_var, cpu_m, cpu_v, cpu_s, cpu_step, p.lr, p.ema_decay, p.beta1, p.beta2, p.eps, p.mode,
            p.bias_correction, p.weight_decay)
        npu_var, npu_m, npu_v, npu_s = npu_apply_fused_ema_adamw(
            npu_grad, npu_var, npu_m, npu_v, npu_s, npu_step, p.lr, p.ema_decay, p.beta1, p.beta2, p.eps, p.mode,
            p.bias_correction, p.weight_decay)
        output = [(npu_var, cpu_var), (npu_m, cpu_m), (npu_v, cpu_v), (npu_s, cpu_s)]
        assert TestNpuApplyFusedEmaAdamW.output_check(output) == 0

    @staticmethod
    def cpu_ema_adamw(grad, var, m, v, s, step,
                      lr, ema_decay, beta1, beta2, eps, mode, bias_correction, weight_decay):
        beta1_correction = 1 - torch.pow(beta1, step) * bias_correction
        beta2_correction = 1 - torch.pow(beta2, step) * bias_correction
        grad_ = grad + weight_decay * var * (1 - mode)
        m_ = beta1 * m + (1 - beta1) * grad_
        v_ = beta2 * v + (1 - beta2) * grad_ * grad_
        next_m = m_ / beta1_correction
        next_v = v_ / beta2_correction
        demon = torch.pow(next_v, 0.5) + eps
        update = next_m / demon + weight_decay * var * mode
        var_ = var - lr * update
        s_ = ema_decay * s + (1 - ema_decay) * var_
        return [var_, m_, v_, s_]

    @staticmethod
    def output_check(output):
        # crit refers to criterion for output absolute error between npu and cpu.
        # criterion varies from different tensor element numbers (using 'num' for short),
        # criterion is 2**(-14) when num is between (0,2048];
        # criterion is 2**(-13) when num is between (2048,16384];
        # criterion is 2**(-12) when num is bigger than 16384;
        # fetching crit from precalculated list with individual rank is more efficient than using 'if' in this case.
        # any inf or nan are unacceptable.
        total = 0
        for npu_out, cpu_out in output:
            num = npu_out.numel()
            rank = (num > 2048) + (num > 16384)
            crit = TestNpuApplyFusedEmaAdamW.crit[rank]
            count = (torch.sum(torch.isnan(npu_out)).item(),
                     torch.sum(torch.isinf(npu_out)).item(),
                     torch.sum(torch.abs(npu_out - cpu_out.npu()) > crit).item())
            total += sum(count)
        return total
