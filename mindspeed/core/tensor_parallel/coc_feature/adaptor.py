from mindspeed.core import training as mc_training

from megatron.training import get_args
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes
)
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_expert_tensor_and_model_parallel_group
)
from megatron.core.tensor_parallel.mappings import (
    _reduce,
    _reduce_scatter_along_first_dim,
    _gather_along_first_dim,
    gather_from_tensor_model_parallel_region
)
from megatron.core.utils import divide

from mindspeed.core.tensor_parallel.coc_feature.coc_column_parallel_linear import CoCColumnParallelLinearImpl
from mindspeed.core.tensor_parallel.coc_feature.coc_row_parallel_linear import CoCRowParallelLinearImpl


class MindSpeedCoCColumnParallelLinear(CoCColumnParallelLinearImpl, ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        if hasattr(kwargs, 'is_expert') and kwargs['is_expert']:
            kwargs['parallel_group'] = get_expert_tensor_and_model_parallel_group()
        else:
            kwargs['parallel_group'] = get_tensor_model_parallel_group()
        kwargs['_initialize_affine_weight_cpu'] = _initialize_affine_weight_cpu
        kwargs['_initialize_affine_weight_gpu'] = _initialize_affine_weight_gpu

        kwargs['get_tensor_model_parallel_group'] = get_tensor_model_parallel_group
        kwargs['get_tensor_model_parallel_world_size'] = get_tensor_model_parallel_world_size
        kwargs['gather_from_tensor_model_parallel_region'] = gather_from_tensor_model_parallel_region
        kwargs['get_tensor_model_parallel_rank'] = get_tensor_model_parallel_rank
        kwargs['set_tensor_model_parallel_attributes'] = set_tensor_model_parallel_attributes

        kwargs['_reduce'] = _reduce
        kwargs['_reduce_scatter_along_first_dim'] = _reduce_scatter_along_first_dim
        kwargs['_gather_along_first_dim'] = _gather_along_first_dim
        kwargs['divide'] = divide
        CoCColumnParallelLinearImpl.__init__(self, *args, **kwargs)


class MindSpeedCoCRowParallelLinear(CoCRowParallelLinearImpl, RowParallelLinear):
    def __init__(self, *args, **kwargs):
        if hasattr(kwargs, 'is_expert') and kwargs['is_expert']:
            kwargs['parallel_group'] = get_expert_tensor_and_model_parallel_group()
        else:
            kwargs['parallel_group'] = get_tensor_model_parallel_group()

        kwargs['_initialize_affine_weight_cpu'] = _initialize_affine_weight_cpu
        kwargs['_initialize_affine_weight_gpu'] = _initialize_affine_weight_gpu

        kwargs['get_tensor_model_parallel_group'] = get_tensor_model_parallel_group
        kwargs['get_tensor_model_parallel_world_size'] = get_tensor_model_parallel_world_size
        kwargs['gather_from_tensor_model_parallel_region'] = gather_from_tensor_model_parallel_region
        kwargs['get_tensor_model_parallel_rank'] = get_tensor_model_parallel_rank

        kwargs['_reduce'] = _reduce
        kwargs['_reduce_scatter_along_first_dim'] = _reduce_scatter_along_first_dim
        kwargs['_gather_along_first_dim'] = _gather_along_first_dim
        kwargs['divide'] = divide
        CoCRowParallelLinearImpl.__init__(self, *args, **kwargs)


