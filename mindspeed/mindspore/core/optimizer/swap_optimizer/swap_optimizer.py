# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from mindspeed.ops.npu_apply_fused_adamw_v2 import npu_apply_fused_adamw_v2
from mindspeed.core.optimizer.swap_optimizer.swap_optimizer import SwapDistributedOptimizer


def opt_states_initialization(self):
    for group in self.shard_fp32_from_float16_groups:
        for main_param in group:
            device_state = self.optimizer.state[main_param]
            cpu_state = self.param_to_cpu_states_map[main_param]
            self.param_to_device_states_map[main_param] = device_state

            amsgrad = self.optimizer.param_to_group_map[main_param]['amsgrad']

            for key in self.state_keys:
                if key in device_state:
                    continue
                if key == 'max_exp_avg_sq' and not amsgrad:
                    device_state[key] = None
                    cpu_state[key] = None
                else:
                    device_state[key] = torch.zeros_like(main_param, memory_format=torch.contiguous_format)
                    cpu_state[key] = torch.empty_like(main_param, device='cpu')
                    cpu_state[key].copy_(device_state[key], non_blocking=True)
                    device_state[key].storage().resize_(0)


@classmethod
def create_tensor_maps(cls, main_param, model_param):
    # optimizer parameter
    cpu_state = {'param': torch.empty_like(main_param, device='cpu')}
    cls.param_to_cpu_states_map[main_param] = cpu_state
    cls.main_param_to_model_param_map[main_param] = model_param
    cls.swap_to_host_events_map[main_param] = None
    cls.copy_to_model_param_events_map[main_param] = None


@classmethod
def swap_tensors_to_device(cls, param):
    if param.storage().size() == 0:
        cpu_state = cls.param_to_cpu_states_map[param]
        param.storage().resize_(cpu_state['param'].nbytes)
        param.copy_(cpu_state['param'], non_blocking=True)

        if param in cls.param_to_device_states_map:
            device_state = cls.param_to_device_states_map[param]
            for key in cls.state_keys:
                if device_state[key] is not None and device_state[key].storage().size() == 0:
                    device_state[key].storage().resize_(cpu_state[key].nbytes)
                    device_state[key].copy_(cpu_state[key], non_blocking=True)

        cls.swap_to_device_events_map[param] = torch.cuda.current_stream().record_event()


def _copy_model_params_to_main_params(self):
    def copy_group_params(model_groups, shard_main_groups):
        for model_group, shard_main_group in zip(model_groups, shard_main_groups):
            for model_param, shard_main_param in zip(model_group, shard_main_group):
                param_range_map = self._get_model_param_range_map(model_param)
                param_range = param_range_map["param"]
                assert param_range.size == shard_main_param.nelement()
                shard_model_param = model_param.view(-1)[param_range.start: param_range.end]
                if shard_main_param.storage().size() != 0:
                    shard_main_param.data.copy_(shard_model_param)
                else:
                    cpu_state = SwapDistributedOptimizer.param_to_cpu_states_map[shard_main_param]
                    shard_main_param.storage().resize_(cpu_state['param'].nbytes)
                    shard_main_param.data.copy_(shard_model_param)
                    cpu_state['param'].copy_(shard_main_param)
                    shard_main_param.storage().resize_(0)
    copy_group_params(self.model_float16_groups, self.shard_fp32_from_float16_groups)
    copy_group_params(self.model_fp32_groups, self.shard_fp32_groups)


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
            group['step'] = torch.tensor([1, ], dtype=torch.int64, device=torch.cuda.current_device())

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
