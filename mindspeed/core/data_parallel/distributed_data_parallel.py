# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Bytedance Inc. All rights reserved.
import logging
from functools import wraps
from collections import deque
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.parallel_state import (
    get_data_parallel_world_size,
    get_data_parallel_group,
    get_tensor_model_parallel_world_size,
    get_global_memory_buffer)
from megatron.legacy.model.transformer import FlashSelfAttention

from megatron.training import get_args
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel, logger
from megatron.core.distributed.param_and_grad_buffer import ParamAndGradBuffer
from megatron.core import parallel_state
from megatron.core.utils import log_single_rank
import torch


@torch.no_grad()
def all_gather_param(param, wait_buffer):
    dp_size = get_data_parallel_world_size()
    group = get_data_parallel_group()
    dim_size = list(param.data.size())
    dim_size[0] = dim_size[0] * dp_size
    param.ds_tensor = param.data
    param.data = torch.empty(dim_size, dtype=param.data.dtype, device=torch.cuda.current_device())
    wait_buffer.append(torch.distributed._all_gather_base(param.data, param.ds_tensor.contiguous(), async_op=True, group=group))


@torch.no_grad()
def reduce_scatter_grad(param, wait_grad_buffer):
    dp_size = get_data_parallel_world_size()
    scale = 1.0
    if dp_size > 0 :
        scale = scale / dp_size
    param.full_grad.data *= scale
    group = get_data_parallel_group()
    param.grad_data_buffer = torch.empty(param.ds_tensor.shape, dtype=param.full_grad.dtype, device=torch.cuda.current_device())
    wait_grad_buffer.append(torch.distributed._reduce_scatter_base(param.grad_data_buffer, param.full_grad.data.contiguous(), async_op=True, group=group))


@torch.no_grad()
def release_param_data(param):
    param.data = param.ds_tensor


def wait_grad(param, wait_grad_buffer):
    wait_grad_buffer.popleft().wait()
    param.main_grad.add_(param.grad_data_buffer)
    param.grad_data_buffer = None
    param.full_grad = None
    param.grad = None


def set_model_fw_bw_hook(modules):
    wait_buffer = deque()
    wait_grad_buffer = deque()
    dp_size = get_data_parallel_world_size()
    if dp_size == 1:
        return 
    module_list = []
    fa_module = False
    for module in modules:
        fa_module |= isinstance(module, FlashSelfAttention)
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
            module.pre_module_id = module.next_module_id = None
            module_list.append(module)
            if fa_module:
                # Send h_to_4h information in advance for communication masking.
                module.light_weight = True
                fa_module = False
    if len(module_list) > 0:
        module_list[0].zero_start = True
        module_list[-1].zero_end = True
    for i in range(len(module_list) - 1):
        module_list[i].next_module_id = i + 1
        module_list[i + 1].pre_module_id = i
    

    def forward_pre_hook(module, *arg):
        if hasattr(module, 'zero_start'):
            all_gather_param(module.weight, wait_buffer)
        wait_buffer.popleft().wait()
        if hasattr(module, 'light_weight'):
            return
        next_module_id = module.next_module_id
        if next_module_id is not None:
            next_module = module_list[next_module_id]
            all_gather_param(next_module.weight, wait_buffer)
            if hasattr(next_module, 'light_weight') and next_module.next_module_id is not None:
                all_gather_param(module_list[next_module.next_module_id].weight, wait_buffer)
        

    def forward_hook(module, *args):
        release_param_data(module.weight)


    def backward_pre_hook(module, *args):
        if hasattr(module, 'zero_end'):
            all_gather_param(module.weight, wait_buffer)
        wait_buffer.popleft().wait()
        if hasattr(module, 'light_weight'):
            return
        pre_module_id = module.pre_module_id
        if pre_module_id is not None:
            pre_module = module_list[pre_module_id]
            all_gather_param(pre_module.weight, wait_buffer)
            if hasattr(pre_module, 'light_weight') and pre_module.pre_module_id is not None:
                all_gather_param(module_list[pre_module.pre_module_id].weight, wait_buffer)


    def backward_hook(module, *arg):
        release_param_data(module.weight)
        reduce_scatter_grad(module.weight, wait_grad_buffer)
        if hasattr(module, 'light_weight'):
            return
        next_module_id = module.next_module_id
        if next_module_id is not None:
            next_module = module_list[next_module_id]
            if hasattr(next_module, 'light_weight') and next_module.next_module_id is not None:
                wait_grad(module_list[next_module.next_module_id].weight, wait_grad_buffer)
            wait_grad(next_module.weight, wait_grad_buffer)
        if hasattr(module, 'zero_start'):
            wait_grad(module.weight, wait_grad_buffer)

    for module in module_list:
        module.register_forward_pre_hook(hook=forward_pre_hook)
        module.register_forward_hook(hook=forward_hook)
        module.register_full_backward_pre_hook(hook=backward_pre_hook)
        module.register_full_backward_hook(hook=backward_hook)


def distributed_data_parallel_init_zero3(
    self,
    config,
    module,
    data_parallel_group,
    accumulate_allreduce_grads_in_fp32: bool,
    overlap_grad_reduce: bool,
    use_distributed_optimizer: bool,
    expert_data_parallel_group,
    disable_bucketing: bool = False,
    check_for_nan_in_grad: bool = False,
    bucket_size: int = 40000000,
):
    super(DistributedDataParallel, self).__init__(config)
    self.module = module
    if get_args().enable_zero3:
        set_model_fw_bw_hook(self.module.modules())

    # Set bucket_size to infinity if overlap_grad_reduce is False.
    self.overlap_grad_reduce = overlap_grad_reduce
    self.use_distributed_optimizer = use_distributed_optimizer

    # Turn off bucketing if overlap_grad_reduce is False, if we are on a pipeline stage
    # that is not the first (since data-parallel communication on these stages is not on
    # the critical path), or if disable_bucketing is True (e.g., we might not want to
    # break up model parameters into buckets for model chunks after the first
    # in the interleaved schedule).
    if not self.overlap_grad_reduce:
        bucket_size = None
    if parallel_state.get_pipeline_model_parallel_rank() > 0:
        bucket_size = None
    if disable_bucketing:
        bucket_size = None

    self.check_for_nan_in_grad = check_for_nan_in_grad
    self.bucket_size = bucket_size

    self.module = module
    self.param_to_buffer = {}
    self.zero3_param = []

    # Group parameters by their gradient type.
    param_to_name = {}
    dense_params = []
    expert_parallel_params = []
    for name, param in self.module.named_parameters():
        if not param.requires_grad:
            continue
        dtype = param.dtype
        param.grad_added_to_main_grad = False
        param_to_name[param] = name

        if hasattr(param, 'enable_zero3') and param.enable_zero3:
            param.main_grad = torch.zeros_like(param, dtype=dtype)
            self.zero3_param.append(param)
            continue

        if getattr(param, 'allreduce', True):
            dense_params.append(param)
        else:
            expert_parallel_params.append(param)
        

    def allocate_buffers_for_parameters(
        input_params, data_parallel_group, gradient_scaling_factor=1.0,
    ):
        param_and_grad_dtype_to_params = {}

        # Group parameters by their gradient type.
        for param in input_params:
            if not param.requires_grad:
                continue

            param_dtype = param.dtype
            grad_dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype

            params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
            params.append(param)
            param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params

        # Allocate the grad buffers and map the grads.
        buffers = []
        for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
            buffers.append(
                ParamAndGradBuffer(
                    param_dtype,
                    grad_dtype,
                    params,
                    data_parallel_group,
                    bucket_size,
                    param_to_name,
                    self.overlap_grad_reduce,
                    self.use_distributed_optimizer,
                    gradient_scaling_factor,
                    self.check_for_nan_in_grad,
                )
            )
            for param in params:
                self.param_to_buffer[param] = buffers[-1]

        return buffers

    data_parallel_world_size = torch.distributed.get_world_size(data_parallel_group)
    # Allocate the param+grad buffers for dense params' grads.
    self.buffers = allocate_buffers_for_parameters(
        dense_params,
        data_parallel_group,
        gradient_scaling_factor=1.0 / data_parallel_world_size,
    )

    # Allocate separate param+grad buffers for expert parallel params' grads.
    self.expert_parallel_buffers = allocate_buffers_for_parameters(
        expert_parallel_params,
        expert_data_parallel_group,
        gradient_scaling_factor=1.0 / data_parallel_world_size,
    )

    # Delete references to weight_tensor if they exist since we don't want two parameter copies
    # if we re-mapped parameters (which happens when we use the distributed optimizer).
    # This is a temporary workaround around a TE bug that is fixed with
    # https://github.com/NVIDIA/TransformerEngine/pull/719.
    if self.use_distributed_optimizer:

        @torch.no_grad()
        def unmap_weight_tensor(m):
            if hasattr(m, 'weight_tensor'):
                m.weight_tensor = None

        self.module.apply(unmap_weight_tensor)

    # Register backward hook.
    # Accumulation function for the gradients need to be stored so they
    # don't go out of scope.
    self.grad_accs = []
    for param in self.module.parameters():
        if param.requires_grad:
            # Expand so we get access to grad_fn.
            param_tmp = param.expand_as(param)
            # Get the gradient accumulator function.
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            if not (hasattr(param, 'enable_zero3') and param.enable_zero3):
                grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))
            self.grad_accs.append(grad_acc)


def distributed_data_parallel_zero_grad_wrapper(function):
    @wraps(function)
    def distributed_data_parallel_zero_grad(self, *args, **kwargs):
        function(self, *args, **kwargs)
        for p in self.zero3_param:
            p.main_grad.data.zero_()
    return distributed_data_parallel_zero_grad


def distributed_data_parallel_init_with_cp(
    self,
    config,
    ddp_config,
    module: torch.nn.Module,
    disable_bucketing: bool = False,
):
    super(DistributedDataParallel, self).__init__(config)
    self.module = module

    # If bucket_size is not provided as an input, use sane default.
    # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
    # ring-reduce implementations are large enough to remain bandwidth-bound rather than
    # latency-bound.
    if ddp_config.bucket_size is None:
        ddp_config.bucket_size = max(
            40000000, 1000000 * parallel_state.get_data_parallel_world_size()
        )
    # Set bucket_size to infinity if overlap_grad_reduce is False.
    if not ddp_config.overlap_grad_reduce:
        ddp_config.bucket_size = None

    self.ddp_config = ddp_config
    log_single_rank(
        logger,
        logging.INFO,
        f'Setting up DistributedDataParallel with config {self.ddp_config}',
    )

    # Turn off bucketing if we are on a pipeline stage that is not the first (since
    # data-parallel communication on these stages is not on the critical path), or if
    # disable_bucketing is True (e.g., we might not want to break up model parameters
    # into buckets for model chunks after the first in the interleaved schedule).
    self.bucket_size = self.ddp_config.bucket_size
    if parallel_state.get_pipeline_model_parallel_rank() > 0:
        self.bucket_size = None
    if disable_bucketing:
        self.bucket_size = None

    self.module = module
    self.param_to_buffer = {}

    # Group parameters by their gradient type.
    param_to_name = {}
    dense_params = []
    expert_parallel_params = []
    for name, param in self.module.named_parameters():
        if not param.requires_grad:
            continue

        param.grad_added_to_main_grad = False
        param_to_name[param] = name

        if getattr(param, 'allreduce', True):
            dense_params.append(param)
        else:
            expert_parallel_params.append(param)

    def allocate_buffers_for_parameters(
        input_params,
        data_parallel_group,
        gradient_scaling_factor,
    ):
        param_and_grad_dtype_to_params = {}

        # Group parameters by their gradient type.
        for param in input_params:
            if not param.requires_grad:
                continue

            param_dtype = param.dtype
            grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype

            params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
            params.append(param)
            param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params

        if not config.calculate_per_token_loss:
            target_gradient_scaling_factor = 1.0 / parallel_state.get_data_parallel_world_size(
                with_context_parallel=True
            )
            if self.ddp_config.average_in_collective:
                # Collective is averaging gradients in collective with data_parallel_group.
                assert (
                    gradient_scaling_factor
                    / torch.distributed.get_world_size(group=data_parallel_group)
                    == target_gradient_scaling_factor
                )
            else:
                assert gradient_scaling_factor == target_gradient_scaling_factor

        # Allocate the grad buffers and map the grads.
        buffers = []
        for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
            buffers.append(
                ParamAndGradBuffer(
                    self.ddp_config,
                    param_dtype,
                    grad_dtype,
                    params,
                    data_parallel_group,
                    self.bucket_size,
                    param_to_name,
                    gradient_scaling_factor,
                )
            )
            for param in params:
                self.param_to_buffer[param] = buffers[-1]

        return buffers

    if config.calculate_per_token_loss:
        gradient_scaling_factor = 1.0
        expert_gradient_scaling_factor = 1.0
    else:
        if self.ddp_config.average_in_collective:
            gradient_scaling_factor = 1.0
            expert_gradient_scaling_factor = (
                1.0 / parallel_state.get_expert_model_parallel_world_size()
            )
        else:
            data_parallel_world_size = parallel_state.get_data_parallel_world_size(
                with_context_parallel=True
            )
            gradient_scaling_factor = 1.0 / data_parallel_world_size
            expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

    # Allocate the param+grad buffers for dense params' grads.
    self.buffers = allocate_buffers_for_parameters(
        dense_params,
        parallel_state.get_data_parallel_group(with_context_parallel=True),
        gradient_scaling_factor=gradient_scaling_factor,
    )

    # Allocate separate param+grad buffers for expert parallel params' grads.
    self.expert_parallel_buffers = allocate_buffers_for_parameters(
        expert_parallel_params,
        parallel_state.get_data_modulo_expert_parallel_group(with_context_parallel=True),
        gradient_scaling_factor=expert_gradient_scaling_factor,
    )

    # Delete references to weight_tensor if they exist since we don't want two parameter copies
    # if we re-mapped parameters (which happens when we use the distributed optimizer).
    # This is a temporary workaround around a TE bug that is fixed with
    if self.ddp_config.use_distributed_optimizer:

        @torch.no_grad()
        def unmap_weight_tensor(m):
            if hasattr(m, 'weight_tensor'):
                m.weight_tensor = None

        self.module.apply(unmap_weight_tensor)

    # Register backward hook.
    # Accumulation function for the gradients need to be stored so they
    # don't go out of scope.
    self.grad_accs = []
    for param in self.module.parameters():
        if param.requires_grad:
            # Expand so we get access to grad_fn.
            param_tmp = param.expand_as(param)
            # Get the gradient accumulator function.
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))
            self.grad_accs.append(grad_acc)
