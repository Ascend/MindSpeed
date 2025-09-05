# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Callable, Optional, List
import warnings

import torch
from torch.nn import functional as F
import torch_npu

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_expert_tensor_parallel_group,
    get_expert_tensor_parallel_rank,
    get_expert_tensor_parallel_world_size,
)
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
    linear_with_grad_accumulation_and_async_allreduce,
)
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
)
from megatron.core.utils import divide


class AttributesBypass:
    def __init__(self, tensor, attrs: List):
        self.attrs = attrs
        self.attrs_value = {}
        self.tensor = tensor
        for key in self.attrs:
            self.attrs_value[key] = getattr(tensor, key, None)

    def __enter__(self):
        if self.tensor is None:
            return
        for key in self.attrs:
            delattr(self.tensor, key)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tensor is None:
            return
        for key in self.attrs:
            setattr(self.tensor, key, self.attrs_value[key])


def load_state_dict_post_hook(weight_keys):
    def hook(module, incompatible_keys):
        full_keys = [k for k in incompatible_keys.missing_keys if any(w in k for w in weight_keys)]
        for k in full_keys:
            incompatible_keys.missing_keys.remove(k)
    return hook


class MindSpeedTELayerNormColumnParallelLinear(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            config,
            init_method: Callable,
            gather_output: bool,
            bias: bool,
            skip_bias_add: bool,
            is_expert: bool = False,
            skip_weight_param_allocation: bool = False,
            tp_comm_buffer_name: Optional[str] = None,
    ):

        super(MindSpeedTELayerNormColumnParallelLinear, self).__init__()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.sequence_parallel = self.config.sequence_parallel
        self.gradient_accumulation_fusion = self.config.gradient_accumulation_fusion

        # MindSpeedTELayerNormColumnParallelLinear check.
        if gather_output:
            raise ValueError('Transformer Engine linear layers do not support gather_output = True')

        # Similar to TE, MoE is currently not supported in MindSpeedTELayerNormColumnParallelLinear.
        if is_expert:
            raise ValueError('Transformer Engine linear layers do not yet support MoE')

        if skip_weight_param_allocation:
            raise ValueError(
                'Transformer Engine linear layers do not support skip_weight_param_allocation'
            )

        # ColumnParallelLine init spec.
        if is_expert:
            # Not should be used for now.
            tp_size = get_expert_tensor_parallel_world_size()
            rank = get_expert_tensor_parallel_rank()
            self.parallel_group = get_expert_tensor_parallel_group()
        else:
            tp_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()
            self.parallel_group = get_tensor_model_parallel_group()

        self.output_size_per_partition = divide(output_size, tp_size)

        self.allreduce_dgrad = (
            tp_size > 1 and not self.sequence_parallel and not self.disable_grad_reduce
        )
        if self.allreduce_dgrad and self.sequence_parallel:
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        # Because skip_weight_param_allocation is not supported in TE, always do weight initialize.
        if config.use_cpu_initialization:
            self.weight = torch.nn.Parameter(
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
                    stride=1,
                    rank=rank,
                    world_size=tp_size,
                )
        else:
            self.weight = torch.nn.Parameter(
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
                    stride=1,
                    is_expert=self.is_expert,
                )

        setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))

        if bias:
            if config.use_cpu_initialization:
                self.bias = torch.nn.Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = torch.nn.Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            # stride=1 in this case.
            set_tensor_model_parallel_attributes(self.bias, True, 0, 1) 
            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
            setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
        else:
            self.register_parameter('bias', None)

        if self.sequence_parallel and tp_size <= 1:
            warnings.warn(
                "`sequence_parallel` is set to `True`, but tensor model parallel size "
                f"is {tp_size}. Disabling sequence parallel."
            )
            self.sequence_parallel = False

        # Forward impl settings without ascend-mc2.
        self._linear_forward_impl = linear_with_grad_accumulation_and_async_allreduce

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f'{prefix}_extra_state'
            )
        )

        # Norm init spec.
        if self.config.normalization not in ['LayerNorm', 'RMSNorm']:
            raise AssertionError('Unsupported normalization type {}!'.format(self.config.normalization))
    
        layer_norm_weight = torch.nn.Parameter(
            torch.ones(self.input_size, device='npu', dtype=self.config.params_dtype)
        )
        self.register_parameter(
            "layer_norm_weight", layer_norm_weight
        )
        setattr(self.layer_norm_weight, 'sequence_parallel', self.sequence_parallel)

        if self.config.normalization != 'RMSNorm':
            layer_norm_bias = torch.nn.Parameter(
                torch.ones(self.input_size, device='npu', dtype=self.config.params_dtype)
            )
            self.register_parameter(
                "layer_norm_bias", layer_norm_bias
            )
            setattr(self.layer_norm_bias, 'sequence_parallel', self.sequence_parallel) 
        else:
            self.layer_norm_bias = None
        
    def _rmsnorm(self, inp):
        if self.config.use_fused_rmsnorm:
            return torch_npu.npu_rms_norm(inp, self.layer_norm_weight, epsilon=self.config.layernorm_epsilon)[0]
        return (inp * torch.rsqrt(inp.pow(2).mean(-1, keepdim=True) + self.config.layernorm_epsilon)) \
                * self.layer_norm_weight

    def forward(self, inp: torch.Tensor, is_first_microbatch: Optional[bool] = None, fp8_output=False):
        if is_first_microbatch is not None or fp8_output is not False:
            raise RuntimeError('{} is not support fp8.'.format(self.__class__.__name__))
        if self.config.normalization == 'LayerNorm':
            norm_output = F.layer_norm(inp, 
                                       weight=self.layer_norm_weight,
                                       bias=self.layer_norm_bias,
                                       eps=self.config.layernorm_epsilon
                                       )
        else:
            norm_output = self._rmsnorm(inp)

        bias = self.bias if not self.skip_bias_add else None

        if (
            self.allreduce_dgrad
            or self.sequence_parallel
            or self.disable_grad_reduce
        ):
            input_parallel = norm_output
        else:
            input_parallel = copy_to_tensor_model_parallel_region(norm_output)

        if not self.config.use_ascend_mc2:
            output_parallel = self._linear_forward_impl(
                input=input_parallel,
                weight=self.weight,
                bias=bias,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                allreduce_dgrad=self.allreduce_dgrad,
                sequence_parallel=self.sequence_parallel,
                grad_output_buffer=(
                    self.grad_output_buffer if self.config.defer_embedding_wgrad_compute else None
                ),
                wgrad_deferral_limit=(
                    self.config.wgrad_deferral_limit
                    if self.config.defer_embedding_wgrad_compute
                    else None
                )
            )
        else:
            from mindspeed.core.tensor_parallel.mc2_feature.linear_function import ColumnSeqParallelLinearFunction \
                as MC2ColumnSeqParallelLinearFunction
            output_parallel = MC2ColumnSeqParallelLinearFunction.apply(
                input_parallel,
                self.weight,
                bias,
                self.parallel_group,
                True,
                self.gradient_accumulation_fusion
            )
        # with TE, gather_output is not supported. Return output_parallel and _linear_bias.
        return output_parallel, bias

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        private_key = [name for name in state.keys() if '_layernorm' in name or '_linear' in name]
        for name in private_key:
            state.pop(name)
        return state

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )