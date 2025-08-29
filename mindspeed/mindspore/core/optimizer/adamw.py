# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch import npu_apply_fused_adamw_v2


def adamw(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          step_tensor: Tensor,
          *,
          amsgrad: bool,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float,
          maximize: bool):
    r"""Functional API that performs AdamW algorithm computation.
    See :class:`~torch.optim.AdamW` for details.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        max_exp_avg_sq = max_exp_avg_sqs[i] if amsgrad else None

        npu_apply_fused_adamw_v2(param,
            grad,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            step_tensor,
            amsgrad=amsgrad,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize)


def step_func(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        state_sums = []
        max_exp_avg_sqs = []
        state_steps = []
        amsgrad = group['amsgrad']
        beta1, beta2 = group['betas']

        if 'step' in group:
            group['step'] += 1
            if group['step'].is_cpu:
                group['step'] = group['step'].cuda()
        else:
            group['step'] = torch.tensor([1, ], dtype=torch.int64, device=torch.cuda.current_device())

        for p in group['params']:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError('AdamW does not support sparse gradients')
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])

            if amsgrad:
                max_exp_avg_sqs.append(state['max_exp_avg_sq'])

        adamw(params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                group['step'],
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'])

    return loss
