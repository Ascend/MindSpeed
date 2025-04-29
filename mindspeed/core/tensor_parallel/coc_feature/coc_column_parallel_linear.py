# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import warnings
from typing import Callable, Optional, List

import torch
from torch.nn import Parameter
import torch.nn.functional as F

from .min_comm_cfg import min_comm_config
from .user_config import get_value_from_cfg

from megatron.core.parallel_state import (
    get_expert_tensor_parallel_rank,
    get_expert_tensor_parallel_world_size,
)


class CoCColumnParallelLinearImpl(torch.nn.Module):

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
        get_tensor_model_parallel_group: Callable = None,
        get_tensor_model_parallel_world_size: Callable = None,
        get_tensor_model_parallel_rank: Callable = None,
        gather_from_tensor_model_parallel_region: Callable = None,
        _reduce: Callable = None,
        _reduce_scatter_along_first_dim: Callable = None,
        _gather_along_first_dim: Callable = None,
        _initialize_affine_weight_cpu: Callable = None,
        _initialize_affine_weight_gpu: Callable = None,
        set_tensor_model_parallel_attributes: Callable = None,
        divide: Callable = None,
    ):
        # Can't use super().__init__() 
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

        if is_expert:
            world_size = get_expert_tensor_parallel_world_size()
            rank = get_expert_tensor_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()
        self.explicit_expert_comm = self.is_expert and (world_size > 1 or self.expert_parallel)

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

        # register once is same used in CoCRowParallelLinearImpl
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

    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None, runtime_gather_output: Optional[bool] = None, **kwargs):
        if min_comm_config.coc_mode == 0 and not min_comm_config.coc_fused_kernel:
            return super().forward(input_, weight, runtime_gather_output)

        bias = self.bias if not self.skip_bias_add else None
        input_parallel = input_
        use_weight = self.weight if weight is None else weight
        if hasattr(self, "norm") and self.norm:
            use_weight = F.normalize(self.weight)
        output_parallel = min_comm_config.column_parallel_function.apply(
            input_parallel,
            use_weight,
            bias
        )
        gather_output = self.gather_output
        # Use the runtime gather output if it's set explicitly.
        if runtime_gather_output is not None:
            gather_output = runtime_gather_output

        if gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = min_comm_config.tp_gather_func(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias