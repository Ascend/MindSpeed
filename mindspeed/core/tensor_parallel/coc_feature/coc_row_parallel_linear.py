# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import warnings
from typing import Callable, Optional, List

import torch
from torch.nn import Parameter

from .min_comm_cfg import min_comm_config
from .user_config import get_value_from_cfg

from megatron.core.parallel_state import (
    get_expert_tensor_parallel_rank,
    get_expert_tensor_parallel_world_size,
)


class CoCRowParallelLinearImpl(torch.nn.Module):

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
        get_tensor_model_parallel_group: Callable = None,
        get_tensor_model_parallel_world_size: Callable = None,
        get_tensor_model_parallel_rank: Callable = None,
        gather_from_tensor_model_parallel_region: Callable = None,
        _reduce: Callable = None,
        _reduce_scatter_along_first_dim: Callable = None,
        _gather_along_first_dim: Callable = None,
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

        # Divide the weight matrix along the last dimension.
        if self.is_expert:
            world_size = get_expert_tensor_parallel_world_size()
            rank = get_expert_tensor_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()
        self.explicit_expert_comm = self.is_expert and (world_size > 1 or self.expert_parallel)

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

        min_comm_config.register_tp_get_functions(get_tensor_model_parallel_group,
                                                    get_tensor_model_parallel_world_size,
                                                    get_tensor_model_parallel_rank,
                                                    gather_from_tensor_model_parallel_region)
        min_comm_config.register_mappings(_reduce,
                                            _reduce_scatter_along_first_dim,
                                            _gather_along_first_dim)
        min_comm_config.register_coc_get_param(
            self.config.coc_mode,
            self.config.coc_parallel_num,
            self.config.coc_fused_kernel
        )
        min_comm_config.register_sequence_parallel_switch(self.sequence_parallel)

        min_comm_config.register_customized_coc(get_value_from_cfg('customized_coc'))
        min_comm_config.register_matmul_soc_friendly_setting(get_value_from_cfg('matmul_soc_friendly'),
                                                                int(get_value_from_cfg('k_min')),
                                                                int(get_value_from_cfg('k_max')))
        min_comm_config.register_all_gather_recomputation_switch(get_value_from_cfg('recompute_all_gather'))
        min_comm_config.register_print_tensor_value_switch(get_value_from_cfg('print_tensor_value_open'))
        min_comm_config.register_column_backward_coc_switch(get_value_from_cfg('enable_coc_in_column_backward'))
        min_comm_config.acquire_module_type(self.config.tensor_model_parallel_size)
        min_comm_config.register_function()

    def forward(self, input_: torch.Tensor, **kwargs):
        if min_comm_config.coc_mode == 0 and not min_comm_config.coc_fused_kernel:
            return super().forward(input_)

        input_parallel = input_
        output_parallel = min_comm_config.row_parallel_function.apply(
            input_parallel,
            self.weight,
            None
        )
        output = output_parallel
        if not self.skip_bias_add:
            output = output + self.bias if self.bias is not None else output
            output_bias = None
        else:
            output_bias = self.bias
        return output, output_bias