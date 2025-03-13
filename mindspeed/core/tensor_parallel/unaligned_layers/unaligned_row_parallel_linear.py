# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from typing import Callable, Optional

import torch
from torch.nn import Parameter

from mindspeed.core.tensor_parallel.unaligned_layers.unaligned_utils import unaligned_divide, \
    unaligned_reduce_scatter_to_sequence_parallel_region, unaligned_linear_with_grad_accumulation_and_async_allreduce


class UnalignedRowParallelLinear(torch.nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            config,
            init_method: Callable,
            bias: bool,
            input_is_parallel: bool,
            skip_bias_add: bool,
            stride: int = 1,
            keep_master_weight_for_test: bool = False,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,  # Not used

            # unaligned parallel arguments
            parallel_group: Optional[torch.distributed.ProcessGroup] = None,
            fusion_number: int = 1,  # the number of linear fused
            seq_length: int = None,
            _initialize_affine_weight_cpu: Callable = None,
            _initialize_affine_weight_gpu: Callable = None,
            linear_with_grad_accumulation_and_async_allreduce=None,
            scatter_to_tensor_model_parallel_region=None,
            linear_with_frozen_weight=None,
            reduce_from_tensor_model_parallel_region=None,
    ):
        torch.nn.Module.__init__(self)

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.config = config
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

        self.explicit_expert_comm = self.is_expert and (
                config.tensor_model_parallel_size > 1 or self.expert_parallel
        )

        # Divide the weight matrix along the last dimension.
        world_size = torch.distributed.get_world_size(group=parallel_group)
        rank = torch.distributed.get_rank(group=parallel_group)

        if self.input_size % fusion_number != 0:
            raise AssertionError('input_size({}) must be divisible by fusion number({})'.format(self.input_size, fusion_number))

        if fusion_number != 1:
            self.input_size_per_partition = unaligned_divide(config.num_query_groups, world_size, rank)
            self.input_size_per_partition *= fusion_number
        else:
            self.input_size_per_partition = unaligned_divide(self.input_size, world_size, rank)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size, self.input_size_per_partition, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.input_size_per_partition,
                    1,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    params_dtype=config.params_dtype,
                    rank=rank,
                    world_size=world_size,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=stride,
                    expert_parallel=(self.is_expert and self.expert_parallel),
                )
        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))

        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )

            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)
        else:
            self.register_parameter('bias', None)

        self._forward_impl = unaligned_linear_with_grad_accumulation_and_async_allreduce

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f'{prefix}_extra_state'
            )
        )

        self.seq_length = seq_length
        self.scatter_to_tensor_model_parallel_region = scatter_to_tensor_model_parallel_region
        self.linear_with_frozen_weight = linear_with_frozen_weight
        self.reduce_from_tensor_model_parallel_region = reduce_from_tensor_model_parallel_region
        self.parallel_group = parallel_group

    def forward(self, input_):
        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                        self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = self.scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        allreduce_dgrad = False
        if not self.weight.requires_grad:
            self._forward_impl = self.linear_with_frozen_weight
            output_parallel = self._forward_impl(
                input=input_parallel,
                weight=self.weight,
                bias=None,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                async_grad_allreduce=allreduce_dgrad,
                sequence_parallel=False,
                grad_output_buffer=None,
                allreduce_dgrad=allreduce_dgrad
            )
        else:
            self._forward_impl = unaligned_linear_with_grad_accumulation_and_async_allreduce
            output_parallel = self._forward_impl(
                input=input_parallel,
                weight=self.weight,
                bias=None,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                sequence_parallel=False,
                grad_output_buffer=None,
                allreduce_dgrad=allreduce_dgrad,
                parallel_group=self.parallel_group,
                seq_length=self.seq_length
            )

        # All-reduce across all the partitions.
        if self.explicit_expert_comm:
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:
            output_ = unaligned_reduce_scatter_to_sequence_parallel_region(output_parallel, self.parallel_group)
        else:
            output_ = self.reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
