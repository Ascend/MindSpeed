# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Dict

import torch
from mindspeed.ops.npu_apply_fused_adamw_v2 import npu_apply_fused_adamw_v2


class SwapParameter:

    def __init__(self, device_param):
        self.device_param = device_param
        self.device_state = None
        self.host_param = torch.empty_like(device_param, pin_memory=True, device='cpu')
        self.host_param.copy_(device_param)
        self.host_state = {
            'param': self.host_param
        }
        self.amsgrad = None
        self.device_param.storage().resize_(0)

    def init_state(self, device_state, amsgrad):
        if 'exp_avg' not in device_state:
            self.device_state = device_state
            self.amsgrad = amsgrad

            device_state['exp_avg'] = torch.zeros_like(self.host_param, memory_format=torch.contiguous_format, device=self.device_param.device)
            device_state['exp_avg_sq'] = torch.zeros_like(self.host_param, memory_format=torch.contiguous_format, device=self.device_param.device)
            if amsgrad:
                device_state['exp_avg_sq'] = torch.zeros_like(self.host_param, memory_format=torch.contiguous_format, device=self.device_param.device)

            self.host_state['exp_avg'] = torch.zeros_like(self.device_param, pin_memory=True, device='cpu')
            self.host_state['exp_avg_sq'] = torch.zeros_like(self.device_param, pin_memory=True, device='cpu')
            if amsgrad:
                self.host_state['max_exp_avg_sq'] = torch.zeros_like(self.device_param, pin_memory=True, device='cpu')

    @staticmethod
    def _swap(src, dest, non_blocking):
        dest.storage().resize_(src.storage().size())
        dest.copy_(src, non_blocking=non_blocking)
        if dest.is_cpu:
            src.storage().resize_(0)

    def swap_to_host(self, non_blocking):
        self._swap(self.device_param, self.host_param, non_blocking)
        self._swap(self.device_state['exp_avg'], self.host_state['exp_avg'], non_blocking)
        self._swap(self.device_state['exp_avg_sq'], self.host_state['exp_avg_sq'], non_blocking)
        if self.amsgrad:
            self._swap(self.device_state['max_exp_avg_sq'], self.host_state['max_exp_avg_sq'], non_blocking)

    def swap_to_device(self, non_blocking):
        self._swap(self.host_param, self.device_param, non_blocking)
        self._swap(self.host_state['exp_avg'], self.device_state['exp_avg'], non_blocking)
        self._swap(self.host_state['exp_avg_sq'], self.device_state['exp_avg_sq'], non_blocking)
        if self.amsgrad:
            self._swap(self.host_state['max_exp_avg_sq'], self.device_state['max_exp_avg_sq'], non_blocking)


class SwapDistributedOptimizerImpl:
    ALL_OPTIMIZER = []

    swap_to_device_stream = None
    swap_to_host_stream = None

    swap_to_device_events_map = {}
    swap_to_host_events_map = {}
    param_to_model_param_map = {}
    copy_to_model_param_events_map = {}

    swap_parameters_map: Dict[torch.Tensor, SwapParameter] = {}
    swap_optimizer_counts: int = 0

    state_keys = ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']

    def __init__(self, optimizer, shard_fp32_from_float16_groups, swap_optimizer_times, swap_optimizer_size):
        self.optimizer = optimizer
        self.shard_fp32_from_float16_groups = shard_fp32_from_float16_groups
        self.swap_optimizer_times = swap_optimizer_times
        self.swap_optimizer_size = int(swap_optimizer_size * 1024 * 1024 * 1024)  # GB > MB > KB > B
        self.is_distributed_optimizer = hasattr(self, 'per_model_buffers')
        self.optimizer.is_swap_optimizer = True
        if SwapDistributedOptimizerImpl.swap_to_device_stream is None:
            SwapDistributedOptimizerImpl.swap_to_device_stream = torch.cuda.Stream()
            SwapDistributedOptimizerImpl.swap_to_host_stream = torch.cuda.Stream()
        SwapDistributedOptimizerImpl.ALL_OPTIMIZER.append(self)

        # create all parameters list for step
        self.reset_param_group()

        # initialization optimizer states
        self.swap_parameter_initialization()

        # print swap parameters sizes
        self.optimizer.swap_numel = self.swap_optimizer_counts // self.swap_optimizer_times
        total_memory = sum([sum([p.numel() * 4 if group['amsgrad'] else p.numel() * 3 for p in group['params']])
                            for group in self.optimizer.param_groups]) * 4 / 1024 / 1024
        swap_memory = self.swap_optimizer_counts / 1024 / 1024
        print('[Rank {}] Swap optimizer: {}/{}(MB)\n'.format(torch.cuda.current_device(), swap_memory, total_memory), end='')

    def swap_parameter_initialization(self):
        for group in self.shard_fp32_from_float16_groups:
            for main_param in group:
                if self.swap_optimizer_counts < self.swap_optimizer_size:
                    amsgrad = self.optimizer.param_to_group_map[main_param]['amsgrad']
                    param_size = main_param.numel() * 16 if amsgrad else main_param.numel() * 12
                    self.swap_optimizer_counts += param_size
                    device_state = self.optimizer.state[main_param]
                    self.swap_parameters_map[main_param] = SwapParameter(main_param)
                    self.swap_parameters_map[main_param].init_state(device_state, amsgrad)

    @classmethod
    def create_tensor_maps(cls, main_param, model_param):
        cls.param_to_model_param_map[main_param] = model_param
        cls.swap_to_host_events_map[main_param] = None
        cls.copy_to_model_param_events_map[main_param] = None

    @classmethod
    def copy_tensor_to_model_param(cls, param):
        if param in cls.swap_parameters_map:
            cls.param_to_model_param_map[param].data.copy_(param)
            cls.copy_to_model_param_events_map[param] = torch.cuda.current_stream().record_event()

    @classmethod
    def wait_copy_to_model_event(cls, param):
        event = cls.copy_to_model_param_events_map[param]
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
            cls.copy_to_model_param_events_map[param] = None

    @classmethod
    def swap_tensors_to_device(cls, param):
        if param in cls.swap_parameters_map:
            cls.swap_parameters_map[param].swap_to_device(non_blocking=True)
            cls.swap_to_device_events_map[param] = torch.cuda.current_stream().record_event()

    @classmethod
    def wait_swap_to_device_event(cls, param):
        event = cls.swap_to_device_events_map.get(param, None)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
            cls.swap_to_device_events_map[param] = None

    @classmethod
    def swap_tensors_to_host(cls, param):
        if param.storage().size() != 0 and param in cls.swap_parameters_map:
            cls.swap_parameters_map[param].swap_to_host(non_blocking=True)
            cls.swap_to_host_events_map[param] = torch.cuda.current_stream().record_event()

    def reset_param_group(self):
        self.optimizer.param_to_group_map = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self.optimizer.param_to_group_map[p] = group


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
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        if 'max_exp_avg_sq' not in state:
            state['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format) if amsgrad else None

        # Swap adamw
        if swap_count == 0:
            torch.cuda.current_stream().wait_stream(SwapDistributedOptimizerImpl.swap_to_host_stream)
            with torch.cuda.stream(SwapDistributedOptimizerImpl.swap_to_device_stream):
                torch.cuda.current_stream().wait_stream(SwapDistributedOptimizerImpl.swap_to_host_stream)
                while i < len(params_list) and (
                        swap_count + params_list[i].numel() <= self.swap_numel or swap_count <= 0):
                    if params_list[i] in SwapDistributedOptimizerImpl.swap_parameters_map:
                        SwapDistributedOptimizerImpl.swap_tensors_to_device(params_list[i])
                        swap_count += params_list[i].numel()
                    i += 1

        SwapDistributedOptimizerImpl.wait_swap_to_device_event(param)
        npu_apply_fused_adamw_v2(param, param.grad, state['exp_avg'], state['exp_avg_sq'], state['max_exp_avg_sq'],
                                 group['step'], group['lr'], beta1, beta2, group['weight_decay'],
                                 group['eps'], amsgrad, group['maximize'])

        SwapDistributedOptimizerImpl.copy_tensor_to_model_param(param)
        with torch.cuda.stream(SwapDistributedOptimizerImpl.swap_to_host_stream):
            SwapDistributedOptimizerImpl.wait_copy_to_model_event(param)
            swap_count -= param.numel()
            SwapDistributedOptimizerImpl.swap_tensors_to_host(param)

    return loss
