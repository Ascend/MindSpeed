# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import warnings
from typing import Callable, Optional, List

import torch
from torch.nn import Parameter

from mindspeed.core.tensor_parallel.unaligned_layers.unaligned_utils import unaligned_divide, \
    unaligned_linear_with_grad_accumulation_and_async_allreduce


class UnalignedColumnParallelLinear(torch.nn.Module):

    def __init__(
            self,
            input_size,
            output_size,
            *,
            config,
            init_method: Callable,
            bias=True,
            gather_output=False,
            stride=1,
            keep_master_weight_for_test=False,
            skip_bias_add=False,
            skip_weight_param_allocation: bool = False,
            embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
            grad_output_buffer: Optional[List[torch.Tensor]] = None,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,  # Not used
            disable_grad_reduce: bool = False,

            # unaligned parallel arguments
            parallel_group: Optional[torch.distributed.ProcessGroup] = None,
            fusion_number: int = 1,  # the number of linear fused
            seq_length: int = None,
            _initialize_affine_weight_cpu: Callable = None,
            _initialize_affine_weight_gpu: Callable = None,
            set_tensor_model_parallel_attributes: Callable = None,
            linear_with_grad_accumulation_and_async_allreduce=None,
            copy_to_tensor_model_parallel_region=None,
            linear_with_frozen_weight=None,
            gather_from_tensor_model_parallel_region=None
    ):
        torch.nn.Module.__init__(self)
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.embedding_activation_buffer = embedding_activation_buffer
        self.grad_output_buffer = grad_output_buffer
        self.config = config
        self.disable_grad_reduce = disable_grad_reduce

        self.explicit_expert_comm = self.is_expert and (
                config.tensor_model_parallel_size > 1 or self.expert_parallel
        )

        world_size = torch.distributed.get_world_size(group=parallel_group)
        rank = torch.distributed.get_rank(group=parallel_group)

        if self.output_size % fusion_number != 0:
            raise AssertionError('output_size({}) must be divisible by fusion number({})'.format(self.output_size, fusion_number))
        if fusion_number != 1:
            self.output_size_per_partition = unaligned_divide(config.num_query_groups, world_size, rank)
            self.output_size_per_partition *= fusion_number
        else:
            self.output_size_per_partition = unaligned_divide(self.output_size, world_size, rank)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:
            if config.use_cpu_initialization:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition, self.input_size, dtype=config.params_dtype
                    )
                )
                if config.perform_initialization:
                    self.master_weight = _initialize_affine_weight_cpu(
                        self.weight,
                        self.output_size,
                        self.input_size,
                        self.output_size_per_partition,
                        0,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        rank=rank,
                        world_size=world_size,
                    )
            else:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        self.input_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
                if config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        expert_parallel=(self.is_expert and self.expert_parallel),
                    )

            setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))
        else:
            self.weight = None

        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
        else:
            self.register_parameter('bias', None)

        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and world_size <= 1:
            warnings.warn(
                f"`sequence_parallel` is set to `True`, but tensor model parallel size is {world_size}. "
                f"Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.allreduce_dgrad = world_size > 1 and not self.sequence_parallel
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

        if self.allreduce_dgrad and self.sequence_parallel:
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f'{prefix}_extra_state'
            )
        )

        self.seq_length = seq_length
        self.copy_to_tensor_model_parallel_region = copy_to_tensor_model_parallel_region
        self.linear_with_frozen_weight = linear_with_frozen_weight
        self.parallel_group = parallel_group
        self.gather_from_tensor_model_parallel_region = gather_from_tensor_model_parallel_region

    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None):
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context == True:
                assert (
                    self.config.cpu_offloading == False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        bias = self.bias if not self.skip_bias_add else None

        if (
            self.allreduce_dgrad
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
        ):
            input_parallel = input_
        else:
            input_parallel = self.copy_to_tensor_model_parallel_region(input_)

        if self.config.defer_embedding_wgrad_compute:
            self.embedding_activation_buffer.append(input_parallel)

        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad
        # Matrix multiply.
        if not weight.requires_grad:
            self._forward_impl = self.linear_with_frozen_weight
            output_parallel = self._forward_impl(
                input=input_parallel,
                weight=weight,
                bias=bias,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                async_grad_allreduce=allreduce_dgrad,
                sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
                grad_output_buffer=self.grad_output_buffer
                if self.config.defer_embedding_wgrad_compute
                else None,
                allreduce_dgrad=allreduce_dgrad
            )
        else:
            self._forward_impl = unaligned_linear_with_grad_accumulation_and_async_allreduce
            output_parallel = self._forward_impl(
                input=input_parallel,
                weight=weight,
                bias=bias,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
                grad_output_buffer=self.grad_output_buffer
                if self.config.defer_embedding_wgrad_compute
                else None,
                allreduce_dgrad=allreduce_dgrad,
                parallel_group=self.parallel_group,
                seq_length=self.seq_length
            )
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = self.gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias