# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Callable, Optional, List

import torch

from mindspeed.core.fusions.fused_rms_norm import RMSNorm
from mindspeed.te.pytorch.module.layernorm import MindSpeedTELayernorm


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
            is_expert: bool,
            skip_weight_param_allocation: bool = False,
            tp_comm_buffer_name: Optional[str] = None,
    ):
        super(MindSpeedTELayerNormColumnParallelLinear, self).__init__()
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear
        self.config = config
        # LayerNorm or RMSNorm
        if self.config.normalization not in ['LayerNorm', 'RMSNorm']:
            raise AssertionError('Unsupported normalization type {}!'.format(self.config.normalization))
        self.layer_norm_bias = None
        if self.config.normalization == 'LayerNorm':
            self._layernorm = MindSpeedTELayernorm(input_size, self.config.layernorm_epsilon, self.config.sequence_parallel)
            self.layer_norm_bias = self._layernorm.bias
        else:
            self._layernorm = RMSNorm(input_size, self.config.layernorm_epsilon, self.config.sequence_parallel, config=self.config)

        with AttributesBypass(self._layernorm.weight, ['sequence_parallel']):
            self.layer_norm_weight = self._layernorm.weight

        # Parallel linear
        self._linear = ColumnParallelLinear(input_size, output_size, config=config, init_method=init_method,
                                            gather_output=gather_output, bias=bias, skip_bias_add=skip_bias_add,
                                            is_expert=is_expert, skip_weight_param_allocation=skip_weight_param_allocation,
                                            tp_comm_buffer_name=tp_comm_buffer_name)

        with AttributesBypass(self._linear.weight, ['tensor_model_parallel', 'partition_dim', 'partition_stride', 'allreduce']):
            self.weight = self._linear.weight
        with AttributesBypass(self._linear.bias, ['tensor_model_parallel', 'partition_dim', 'partition_stride', 'allreduce']):
            self.bias = self._linear.bias

        self.register_load_state_dict_post_hook(load_state_dict_post_hook(['_layernorm.weight', '_linear.weight']))

    def forward(self, inp: torch.Tensor, is_first_microbatch: Optional[bool] = None, fp8_output=False):
        if is_first_microbatch is not None or fp8_output is not False:
            raise RuntimeError('{} is not support fp8.'.format(self.__class__.__name__))
        return self._linear(self._layernorm(inp))

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