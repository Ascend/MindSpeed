# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Dict
import logging

from megatron.core.distributed.distributed_data_parallel import logger

from megatron.core import parallel_state

from megatron.core.utils import log_single_rank

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import ParamAndGradBuffer

from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
import torch


def distributed_data_parallel_init(
    self,
    config: TransformerConfig,
    ddp_config: DistributedDataParallelConfig,
    module: torch.nn.Module,
    disable_bucketing: bool = False,
):
    super(DistributedDataParallel, self).__init__(config=config)
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
            target_gradient_scaling_factor = 1.0 / parallel_state.get_data_parallel_world_size()
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
            data_parallel_world_size = parallel_state.get_data_parallel_world_size()
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
    # https://github.com/NVIDIA/TransformerEngine/pull/719.
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

            # Get the gradient accumulator function.
            param.register_hook(self._make_param_hook(param, self.param_to_buffer))


def local_make_param_hook( # change name
    self,
    param: torch.nn.Parameter,
    param_to_buffer: Dict[torch.nn.Parameter, ParamAndGradBuffer],
):
    """
    Creates the all-reduce / reduce-scatter hook for backprop.
    """

    def param_hook(*unused):
        if param.requires_grad:
            if self.ddp_config.overlap_grad_reduce:
                assert (
                    param.grad is not None
                ), 'param.grad being None is not safe when overlap_grad_reduce is True'
            if param.grad is not None and (
                not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
            ):
                param.main_grad.add_(*unused)
            param.grad = None

            if self.ddp_config.overlap_grad_reduce:
                param_to_buffer[param].register_grad_ready(param)

            if hasattr(param, "main_grad"):
                return param.main_grad
            return param.grad

    return param_hook
