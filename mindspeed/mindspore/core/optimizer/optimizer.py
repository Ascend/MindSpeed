# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Callable

import torch

from megatron.core.optimizer.optimizer_config import OptimizerConfig


def megatron_optimizer_init(
    self,
    optimizer: torch.optim.Optimizer,
    config: OptimizerConfig,
    init_state_fn: Callable = lambda x: None,
):
    """Input optimizer is the base optimizer (e.g., Adam)."""
    self.optimizer = optimizer
    assert self.optimizer, 'no optimizer is provided.'
    self.empty_optmizer = False
    if getattr(self.optimizer.param_groups[0]['params'][0], 'fake', False):
        self.empty_optmizer = True
    self.config = config
    self.init_state_fn = init_state_fn


def step(self, closure=None):
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


def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
    """Simple scaling."""
    return loss