# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import os
import types
import copy
import warnings
import random
import numpy
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as functional
import pytest

from mindspeed.core.optimizer.fused_ema_adamw.fused_ema_adamw import FusedEmaAdamW

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestFusedEmaAdamW:
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(functional.relu(self.conv1(x)))
            x = self.pool(functional.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = functional.relu(self.fc1(x))
            x = functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    @pytest.mark.skip(reason='temporary skip for compatibility with released CANN.')
    def test_optimizer(self):
        TestFusedEmaAdamW.seed_all(mode=True)
        base_net = TestFusedEmaAdamW.Net().to(torch.float32).npu()
        nets = [base_net, copy.deepcopy(base_net)]
        optimizers = [FusedEmaAdamW(nets[0].parameters()), FusedEmaAdamW(nets[1].parameters())]
        optimizers[0].step = types.MethodType(TestFusedEmaAdamW.base_step, optimizers[0])
        data = [(i, torch.randint(1, 10, (1, 3, 32, 32)).to(torch.float32).npu()) for i in range(10)]
        for i, data_ in data:
            step_loss = []
            for net, optimizer in zip(nets, optimizers):
                optimizer.zero_grad()
                output = net(data_)
                step_loss.append(output.sum())
                output.sum().backward()
                optimizer.step()
            if i == 0:
                Checker.compair_optimizer(optimizers[0], optimizers[1])
            ae = torch.abs(step_loss[1] - step_loss[0]).item()
            re = torch.div(ae, step_loss[0].add(1e-7)).item()
            assert re < 0.005

    @staticmethod
    @torch.no_grad()
    def base_step(self, closure=None):
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
            beta1, beta2 = group['betas']
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = torch.tensor([int(1)]).npu()
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW dose not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['ema_params'] = p.data.clone()
                bias_correction1 = beta1 ** (group['step'].item() - 1)
                bias_correction2 = beta2 ** (group['step'].item() - 1)
                p.data, state['exp_avg'], state['exp_avg_sq'] = torch_npu.npu_apply_adam_w(
                    bias_correction1, bias_correction2, group['lr'], group['weight_decay'], beta1, beta2, group['eps'],
                    p.grad, None, False, False, out=(p.data, state['exp_avg'], state['exp_avg_sq']))
                state['ema_params'].mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
        return loss

    @staticmethod
    def seed_all(seed=1234, mode=False):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(mode)
        torch_npu.npu.manual_seed_all(seed)


class Checker:
    crit = [2 ** (-14), 2 ** (-13), 2 ** (-12)]

    @staticmethod
    def compair_optimizer(base_optimizer, new_optimizer):
        for base_group, new_group in zip(base_optimizer.param_groups, new_optimizer.param_groups):
            for base_p, new_p in zip(base_group['params'], new_group['params']):
                if base_p.grad is None:
                    continue
                base_state, new_state = base_optimizer.state[base_p], new_optimizer.state[new_p]
                Checker.compair_tensor(base_p, new_p)
                Checker.compair_tensor(base_state['exp_avg'], new_state['exp_avg'])
                Checker.compair_tensor(base_state['exp_avg_sq'], new_state['exp_avg_sq'])
                Checker.compair_tensor(base_state['ema_params'], new_state['ema_params'])

    @staticmethod
    def compair_tensor(base, new):
        # crit refers to criterion for output errors(absolute error,relative error,
        # max relative error, mean relative error, root-mean-square error) between npu and cpu.
        # criterion varies from different tensor element numbers (using 'num' for short),
        # criterion is 2**(-14) when num is between (0,2048];
        # criterion is 2**(-13) when num is between (2048,16384];
        # criterion is 2**(-12) when num is bigger than 16384;
        # fetching crit from precalculated list with individual rank is more efficient than using 'if' in this case.
        num = torch.numel(base)
        ae = torch.abs(new - base)
        re = torch.div(ae, base.add(1e-7))
        rank = (num > 2048) + (num > 16384)
        crit = Checker.crit[rank]
        count = not (torch.sum(ae > crit).item() + torch.sum(re > crit).item())
        mare = torch.max(re).item()
        mere = torch.mean(re).item()
        rmse = torch.sqrt(torch.div(torch.sum(torch.pow(ae, 2)), num)).item()
        sub_check = (mare < crit) * (mere < crit) * (rmse < crit) * count
        assert sub_check
