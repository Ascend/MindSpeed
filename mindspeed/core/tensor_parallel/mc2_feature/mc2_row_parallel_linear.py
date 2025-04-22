# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import warnings
from typing import Callable, Optional, List

import torch
from torch.nn import Parameter

from .linear_function import RowSeqParallelLinearFunction, RowSeqParallelLinearWithFrozenWeightFunction


class MC2RowParallelLinearImpl(torch.nn.Module):
    """
    MC2 splits larger computation and communication tasks into smaller subtasks
    and pipelines them in such a way that communication subtasks and computation subtasks can overlap with each other.
    This reduces waiting time and improves overall utilization efficiency.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X
    along its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]
    It employs a fused ops to combine matmul computations with reduce_scatter in forward and all_gather in backward.
    """
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

        # coc parallel arguments
        parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        _initialize_affine_weight_cpu: Callable = None,
        _initialize_affine_weight_gpu: Callable = None,
        divide: Callable = None,
        get_tensor_model_parallel_world_size: Callable = None,
        get_tensor_model_parallel_group: Callable = None,
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

        world_size = torch.distributed.get_world_size(group=parallel_group)
        rank = torch.distributed.get_rank(group=parallel_group)

        self.input_size_per_partition = divide(input_size, world_size)

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
                    is_expert=self.is_expert,
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

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f'{prefix}_extra_state'
            )
        )

        self.parallel_group = parallel_group

    def forward(self, input_: torch.Tensor, **kwargs):
        if not self.weight.requires_grad:
            output = RowSeqParallelLinearWithFrozenWeightFunction.apply(
                input_, self.weight, None, self.parallel_group
            )
        else:
            output = RowSeqParallelLinearFunction.apply(
                input_, self.weight, None, self.parallel_group, self.gradient_accumulation_fusion
            )

        if not self.skip_bias_add:
            output = output + self.bias if self.bias is not None else output
            output_bias = None
        else:
            output_bias = self.bias

        return output, output_bias