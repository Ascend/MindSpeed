# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu
)
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_expert_tensor_and_model_parallel_group
)
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.utils import divide
from mindspeed.core.tensor_parallel.mc2_feature.mc2_column_parallel_linear import MC2ColumnParallelLinearImpl
from mindspeed.core.tensor_parallel.mc2_feature.mc2_row_parallel_linear import MC2RowParallelLinearImpl


def _resolve_tp_group(kwargs):
    """Use the explicit MCore tp_group when present.

    MC2's autograd functions carry the resolved group per layer, so a supplied
    ``tp_group`` does not need to fall back to global parallel state.
    """
    if 'parallel_group' in kwargs:
        raise TypeError("parallel_group is not supported; use the Megatron tp_group argument")

    tp_group = kwargs.get('tp_group')
    if tp_group is not None:
        return tp_group
    if kwargs.get('is_expert', False):
        return get_expert_tensor_and_model_parallel_group()
    return get_tensor_model_parallel_group()


class MindSpeedMC2ColumnParallelLinear(MC2ColumnParallelLinearImpl, ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        kwargs['tp_group'] = _resolve_tp_group(kwargs)
        kwargs['_initialize_affine_weight_cpu'] = _initialize_affine_weight_cpu
        kwargs['_initialize_affine_weight_gpu'] = _initialize_affine_weight_gpu

        kwargs['gather_from_tensor_model_parallel_region'] = gather_from_tensor_model_parallel_region
        kwargs['set_tensor_model_parallel_attributes'] = set_tensor_model_parallel_attributes
        kwargs['divide'] = divide
        MC2ColumnParallelLinearImpl.__init__(self, *args, **kwargs)


class MindSpeedMC2RowParallelLinear(MC2RowParallelLinearImpl, RowParallelLinear):
    def __init__(self, *args, **kwargs):
        kwargs['tp_group'] = _resolve_tp_group(kwargs)

        kwargs['_initialize_affine_weight_cpu'] = _initialize_affine_weight_cpu
        kwargs['_initialize_affine_weight_gpu'] = _initialize_affine_weight_gpu
        kwargs['divide'] = divide
        MC2RowParallelLinearImpl.__init__(self, *args, **kwargs)
