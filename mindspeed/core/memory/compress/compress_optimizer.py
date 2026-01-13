# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from typing import List  # codecheck_ignore

import torch
import torch.distributed
import torch_npu
from torch import Tensor

from .utils import TensorManager, ShareMemory


class CompressTensor(TensorManager):
    """ Class for managing first and second-order momentum 
    compression and decompression states.
    """
    def __init__(self, like_tensor: torch.Tensor, compress_ratio: float = 0.5) -> None:
        if compress_ratio < 0 or compress_ratio > 1:
            raise ValueError
        super().__init__(like_tensor, compress_ratio)
        self.normal_init(like_tensor) # replace tensor
        self.filter_condition = self.can_be_compress(like_tensor)
        if self.filter_condition:
            self.compress_init(like_tensor)
        self.shape, self.dtype, self.device = like_tensor.shape, like_tensor.dtype, like_tensor.device
    
    def normal_init(self, like_tensor: torch.Tensor):
        self.tensor = torch.zeros(like_tensor.shape, device=like_tensor.device, dtype=like_tensor.dtype)
    
    def compress_init(self, like_tensor: torch.Tensor):
        var = ShareMemory(numel=like_tensor.numel() // 2, dtype=like_tensor.dtype)
        var.can_be_used = False
        self.malloc(var, statistic=True)
        self.encode()

    def encode_state(self):
        self.encode()
        self.tensor = None
    
    def decode_state(self):
        self.tensor = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        self.decode()
    
    def recover(self):
        if self.filter_condition:
            self.decode_state()
        return self.tensor
    
    def update(self, step):
        if self.filter_condition:
            self.encode_state()
            self.adjust_pdf_statistic(step)
    
    def adjust_pdf_statistic(self, step):
        if step < 3 or step % 100 == 0:
            self.statistic = True
        else:
            self.statistic = False
    
    @staticmethod
    def can_be_compress(tensor: torch.Tensor):
        return tensor.numel() % 64 == 0 and tensor.numel() > 32768


def compress_adamw_impl(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[CompressTensor],
          exp_avg_sqs: List[CompressTensor],
          max_exp_avg_sqs: List[Tensor],
          step: int,
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
        exp_avg = exp_avgs[i].recover()
        exp_avg_sq = exp_avg_sqs[i].recover()

        # Perform stepweight decay; param.mul_(1 - lr * weight_decay)
        bias_correction1 = beta1 ** (step - 1)
        bias_correction2 = beta2 ** (step - 1)

        param.data, exp_avg, exp_avg_sq = torch_npu.npu_apply_adam_w(
            bias_correction1,
            bias_correction2,
            lr,
            weight_decay,
            beta1,
            beta2,
            eps,
            grad,
            None,
            amsgrad,
            maximize,
            out=(param.data, exp_avg, exp_avg_sq)
        )
        exp_avgs[i].update(step)
        exp_avg_sqs[i].update(step)


def compress_optimizer_step_impl(self, closure=None):
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
        else:
            group['step'] = 1

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
                state['exp_avg'] = CompressTensor(p)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = CompressTensor(p, 0.4)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])

            if amsgrad:
                max_exp_avg_sqs.append(state['max_exp_avg_sq'])


        compress_adamw_impl(params_with_grad,
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
