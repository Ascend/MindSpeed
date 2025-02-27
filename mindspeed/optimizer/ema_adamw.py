from typing import List, Optional
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from mindspeed.ops.npu_apply_fused_ema_adamw import npu_apply_fused_ema_adamw


def fused_ema_adamw(grad: List[Tensor],
                    var: List[Tensor],
                    m: List[Tensor],
                    v: List[Tensor],
                    s: List[Tensor],
                    step: Tensor,
                    lr: float,
                    ema_decay: float,
                    beta1: float,
                    beta2: float,
                    eps: float,
                    mode: int,
                    bias_correction: bool,
                    weight_decay: float):
    for i, param in enumerate(var):
        g_ref = grad[i]
        m_ref = m[i]
        v_ref = v[i]
        s_ref = s[i]
        param.data, m_ref, v_ref, s_ref = npu_apply_fused_ema_adamw(g_ref,
                                                                    param.data,
                                                                    m_ref,
                                                                    v_ref,
                                                                    s_ref,
                                                                    step,
                                                                    lr,
                                                                    ema_decay,
                                                                    beta1,
                                                                    beta2,
                                                                    eps,
                                                                    mode,
                                                                    bias_correction,
                                                                    weight_decay)


class FusedEmaAdamW(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 eps=1e-8,
                 betas=(0.9, 0.999),
                 weight_decay=1e-2,
                 ema_decay=0.9999,
                 amsgrad=False,
                 *,
                 maximize=False,
                 use_num_updates=True,
                 bias_correction=True,
                 adam_w_mode=True,
                 set_grad_none=True
                 ):

        if amsgrad:
            raise RuntimeError('ema_adamw does not support the AMSGrad variant.')
        if maximize:
            raise RuntimeError('ema_adamw does not support the maximize variant.')
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr,
                        bias_correction=bias_correction,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        amsgrad=amsgrad,
                        maximize=maximize)
        super(FusedEmaAdamW, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none
        self.ema_decay = ema_decay
        if use_num_updates:
            self.num_updates = 0
        else:
            self.num_updates = -1

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedEmaAdamW, self).zero_grad()

    def copy_to(self):
        for group in self.param_groups:
            if len(group['params']) == 0:
                continue
            for p in group['params']:
                state = self.state[p]
                if 'ema_params' not in state.keys():
                    continue
                p.data.copy_(state['ema_params'].data)

    def store(self, parameters):
        self.collected_params_group = []
        for group in parameters:
            if len(group['params']) == 0:
                continue
            collected_params = [param.detach().cpu().clone() for param in group['params']]
            self.collected_params_group.append(collected_params)

    def restore(self, parameters):
        for c_group, group in zip(self.collected_params_group, parameters):
            if len(group['params']) == 0:
                continue
            for c_param, param in zip(c_group, group['params']):
                param.data.copy_(c_param.data)
        del self.collected_params_group

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        ema_decay = self.ema_decay
        if self.num_updates >= 0:
            self.num_updates += 1
            ema_decay = min(self.ema_decay, (1 + self.num_updates) / (10 + self.num_updates))
        for group in self.param_groups:
            if len(group['params']) == 0:
                continue
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            ema_params = []
            beta1, beta2 = group['betas']
            bias_correction = True if group['bias_correction'] else False
            valid_dtype = [torch.float32, torch.float16, torch.bfloat16]

            if 'step' in group:
                if not group['step'].is_npu:
                    group['step'] = group['step'].npu()
                group['step'] += 1
            else:
                group['step'] = torch.tensor([int(1)]).npu()

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.dtype not in valid_dtype:
                    raise RuntimeError('ema_adamw only support fp32, fp16, bf16.')
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
                    state['ema_params'] = p.data.clone()
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                ema_params.append(state['ema_params'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

            fused_ema_adamw(grads,
                            params_with_grad,
                            exp_avgs,
                            exp_avg_sqs,
                            ema_params,
                            group['step'],
                            group['lr'],
                            ema_decay,
                            beta1,
                            beta2,
                            group['eps'],
                            self.adam_w_mode,
                            bias_correction,
                            group['weight_decay'])
        return loss
