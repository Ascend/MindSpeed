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
from megatron.core.tensor_parallel.mappings import (
    _reduce,
    _reduce_scatter_along_first_dim,
    _gather_along_first_dim,
    gather_from_tensor_model_parallel_region
)
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.utils import divide
from mindspeed.core.tensor_parallel.mc2_feature.mc2_column_parallel_linear import MC2ColumnParallelLinearImpl
from mindspeed.core.tensor_parallel.mc2_feature.mc2_row_parallel_linear import MC2RowParallelLinearImpl


class MindSpeedMC2ColumnParallelLinear(MC2ColumnParallelLinearImpl, ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        if getattr(kwargs, 'is_expert', False):
            kwargs['parallel_group'] = get_expert_tensor_and_model_parallel_group()
        else:
            kwargs['parallel_group'] = get_tensor_model_parallel_group()
        kwargs['_initialize_affine_weight_cpu'] = _initialize_affine_weight_cpu
        kwargs['_initialize_affine_weight_gpu'] = _initialize_affine_weight_gpu

        kwargs['gather_from_tensor_model_parallel_region'] = gather_from_tensor_model_parallel_region
        kwargs['set_tensor_model_parallel_attributes'] = set_tensor_model_parallel_attributes
        kwargs['divide'] = divide
        MC2ColumnParallelLinearImpl.__init__(self, *args, **kwargs)


class MindSpeedMC2RowParallelLinear(MC2RowParallelLinearImpl, RowParallelLinear):
    def __init__(self, *args, **kwargs):
        if getattr(kwargs, 'is_expert', False):
            kwargs['parallel_group'] = get_expert_tensor_and_model_parallel_group()
        else:
            kwargs['parallel_group'] = get_tensor_model_parallel_group()

        kwargs['_initialize_affine_weight_cpu'] = _initialize_affine_weight_cpu
        kwargs['_initialize_affine_weight_gpu'] = _initialize_affine_weight_gpu
        kwargs['divide'] = divide
        MC2RowParallelLinearImpl.__init__(self, *args, **kwargs)
