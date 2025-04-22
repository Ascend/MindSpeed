# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import warnings
from typing import Callable, Optional, List

import torch
from torch.nn import Parameter

from .linear_function import ColumnSeqParallelLinearFunction, ColumnSeqParallelLinearWithFrozenWeightFunction


class MC2ColumnParallelLinearImpl(torch.nn.Module):
    """
    It splits larger computation and communication tasks into smaller subtasks
    and pipelines them in such a way that communication subtasks and computation subtasks can overlap with each other.
    This reduces waiting time and improves overall utilization efficiency.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    It employs a fused ops approach to combine matmul computations with all_gather in forward.
    """
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

        # coc parallel arguments
        parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        gather_from_tensor_model_parallel_region: Callable = None,
        _initialize_affine_weight_cpu: Callable = None,
        _initialize_affine_weight_gpu: Callable = None,
        set_tensor_model_parallel_attributes: Callable = None,
        divide: Callable = None,
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

        world_size = torch.distributed.get_world_size(group=parallel_group)
        rank = torch.distributed.get_rank(group=parallel_group)

        self.output_size_per_partition = divide(output_size, world_size)

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
                        is_expert=self.is_expert,
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
                "`sequence_parallel` is set to `True`, but tensor model parallel size "
                f"is {world_size}. Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.allreduce_dgrad = (
                world_size > 1 and not self.sequence_parallel and not self.disable_grad_reduce
        )

        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

        if self.allreduce_dgrad and self.sequence_parallel:
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f'{prefix}_extra_state'
            )
        )

        # register some param and func to use in global
        # register once is same used in MC2RowParallelLinearImpl
        self.gather_from_tensor_model_parallel_region = gather_from_tensor_model_parallel_region
        self.parallel_group = parallel_group

    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None, runtime_gather_output: Optional[bool] = None, **kwargs):
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass"
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)},"
                    f"not {expected_shape} as expected"
                )

        bias = self.bias if not self.skip_bias_add else None

        if not weight.requires_grad:
            output = ColumnSeqParallelLinearWithFrozenWeightFunction.apply(
                input_, weight, bias, self.parallel_group, True
            )
        else:
            output = ColumnSeqParallelLinearFunction.apply(
                input_, weight, bias, self.parallel_group, True, self.gradient_accumulation_fusion
            )

        gather_output = self.gather_output
        # Use the runtime gather output if it's set explicitly.
        if runtime_gather_output is not None:
            gather_output = runtime_gather_output

        if gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = self.gather_from_tensor_model_parallel_region(output)
        else:
            output = output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias