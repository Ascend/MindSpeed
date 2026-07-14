import torch

from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_expert_tensor_parallel_group,
    get_expert_tensor_parallel_world_size,
    get_expert_tensor_parallel_rank,
)
from megatron.core.tensor_parallel.mappings import (
    _reduce,
    _reduce_scatter_along_first_dim,
    _gather_along_first_dim,
    gather_from_tensor_model_parallel_region,
)
from megatron.core.utils import divide

from mindspeed.core.tensor_parallel.coc_feature.coc_column_parallel_linear import CoCColumnParallelLinearImpl
from mindspeed.core.tensor_parallel.coc_feature.coc_row_parallel_linear import CoCRowParallelLinearImpl


def _resolve_tp_group(kwargs):
    """Keep the caller's explicit process-group contract intact.

    The Megatron 0.17 linear interface accepts ``tp_group``.  CoC keeps a
    process-global execution configuration, so its optimized path is safe only
    for the default non-expert TP group.  A custom group (and every expert
    layer) uses the MCore fallback, whose per-layer ``self.tp_group`` preserves
    the caller's communication domain.
    """
    if 'parallel_group' in kwargs:
        raise TypeError("parallel_group is not supported; use the Megatron tp_group argument")

    tp_group = kwargs.get('tp_group')

    is_expert = kwargs.get('is_expert', False)
    if tp_group is not None and not is_expert:
        # Megatron 0.17 passes pg_collection.tp explicitly even for the normal
        # default TP domain.  Preserve CoC in that case without requiring MPU
        # to be initialized for genuinely custom groups.
        default_group = get_tensor_model_parallel_group(check_initialized=False)
        if default_group is not None and tp_group is default_group:
            return (
                tp_group,
                True,
                get_tensor_model_parallel_group,
                get_tensor_model_parallel_world_size,
                get_tensor_model_parallel_rank,
            )

    if tp_group is not None:
        def group_getter():
            return tp_group

        def world_size_getter():
            return torch.distributed.get_world_size(group=tp_group)

        def rank_getter():
            return torch.distributed.get_rank(group=tp_group)

        return tp_group, False, group_getter, world_size_getter, rank_getter

    if is_expert:
        tp_group = get_expert_tensor_parallel_group()
        group_getter = get_expert_tensor_parallel_group
        world_size_getter = get_expert_tensor_parallel_world_size
        rank_getter = get_expert_tensor_parallel_rank
    else:
        tp_group = get_tensor_model_parallel_group()
        group_getter = get_tensor_model_parallel_group
        world_size_getter = get_tensor_model_parallel_world_size
        rank_getter = get_tensor_model_parallel_rank

    # The optimized CoC autograd functions read min_comm_config.  Do not let a
    # custom or expert layer overwrite that process-global state.
    use_coc = not is_expert
    return tp_group, use_coc, group_getter, world_size_getter, rank_getter


class MindSpeedCoCColumnParallelLinear(CoCColumnParallelLinearImpl, ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        (
            kwargs['tp_group'],
            kwargs['use_coc'],
            get_parallel_group,
            get_parallel_world_size,
            get_parallel_rank,
        ) = _resolve_tp_group(kwargs)
        kwargs['_initialize_affine_weight_cpu'] = _initialize_affine_weight_cpu
        kwargs['_initialize_affine_weight_gpu'] = _initialize_affine_weight_gpu

        kwargs['get_tensor_model_parallel_group'] = get_parallel_group
        kwargs['get_tensor_model_parallel_world_size'] = get_parallel_world_size
        kwargs['gather_from_tensor_model_parallel_region'] = gather_from_tensor_model_parallel_region
        kwargs['get_tensor_model_parallel_rank'] = get_parallel_rank
        kwargs['set_tensor_model_parallel_attributes'] = set_tensor_model_parallel_attributes

        kwargs['_reduce'] = _reduce
        kwargs['_reduce_scatter_along_first_dim'] = _reduce_scatter_along_first_dim
        kwargs['_gather_along_first_dim'] = _gather_along_first_dim
        kwargs['divide'] = divide
        CoCColumnParallelLinearImpl.__init__(self, *args, **kwargs)


class MindSpeedCoCRowParallelLinear(CoCRowParallelLinearImpl, RowParallelLinear):
    def __init__(self, *args, **kwargs):
        (
            kwargs['tp_group'],
            kwargs['use_coc'],
            get_parallel_group,
            get_parallel_world_size,
            get_parallel_rank,
        ) = _resolve_tp_group(kwargs)

        kwargs['_initialize_affine_weight_cpu'] = _initialize_affine_weight_cpu
        kwargs['_initialize_affine_weight_gpu'] = _initialize_affine_weight_gpu

        kwargs['get_tensor_model_parallel_group'] = get_parallel_group
        kwargs['get_tensor_model_parallel_world_size'] = get_parallel_world_size
        kwargs['gather_from_tensor_model_parallel_region'] = gather_from_tensor_model_parallel_region
        kwargs['get_tensor_model_parallel_rank'] = get_parallel_rank

        kwargs['_reduce'] = _reduce
        kwargs['_reduce_scatter_along_first_dim'] = _reduce_scatter_along_first_dim
        kwargs['_gather_along_first_dim'] = _gather_along_first_dim
        kwargs['divide'] = divide
        CoCRowParallelLinearImpl.__init__(self, *args, **kwargs)
