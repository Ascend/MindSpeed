# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from mindspeed.ops.npu_apply_fused_adamw_v2 import npu_apply_fused_adamw_v2
from mindspeed.core.optimizer.swap_optimizer.swap_optimizer import SwapDistributedOptimizer


def swap_adamw_step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        if 'step' in group:
            group['step'] += 1
            if group['step'].is_cpu:
                group['step'] = group['step'].cuda()
        else:
            group['step'] = torch.tensor(1, dtype=torch.int64, device=torch.cuda.current_device())

    swap_count = 0
    params_list = list(self.param_to_group_map.keys())
    for i, param in enumerate(params_list):
        if param.grad is None:
            continue
        if param.grad.is_sparse:
            raise RuntimeError('AdamW does not support sparse gradients')

        group = self.param_to_group_map[param]
        amsgrad = group['amsgrad']
        beta1, beta2 = group['betas']
        state = self.state[param]

        # State initialization
        if len(state) == 0:
            state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        if 'max_exp_avg_sq' not in state:
            state['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format) if amsgrad else None

        # Swap adamw
        if swap_count == 0:
            torch.cuda.current_stream().wait_stream(SwapDistributedOptimizer.swap_to_host_stream)
            with torch.cuda.stream(SwapDistributedOptimizer.swap_to_device_stream):
                if i == 0:
                    torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
                torch.cuda.current_stream().wait_stream(SwapDistributedOptimizer.swap_to_host_stream)
                while i < len(params_list) and (swap_count + params_list[i].numel() <= self.swap_numel or swap_count <= 0):
                    SwapDistributedOptimizer.swap_tensors_to_device(params_list[i])
                    swap_count += params_list[i].numel()
                    i += 1

        SwapDistributedOptimizer.wait_swap_to_device_event(param)
        npu_apply_fused_adamw_v2(param, param.grad, state['exp_avg'], state['exp_avg_sq'], state['max_exp_avg_sq'],
                                 group['step'], group['lr'], beta1, beta2, group['weight_decay'],
                                 group['eps'], amsgrad, group['maximize'])

        SwapDistributedOptimizer.copy_tensor_to_model_param(param)
        with torch.cuda.stream(SwapDistributedOptimizer.swap_to_host_stream):
            SwapDistributedOptimizer.wait_copy_to_model_event(param)
            swap_count -= param.numel()
            SwapDistributedOptimizer.swap_tensors_to_host(param)

    return loss
