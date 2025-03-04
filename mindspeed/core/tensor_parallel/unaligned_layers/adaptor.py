# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
from megatron.training import get_args
from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_and_expert_parallel_group
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, _initialize_affine_weight_cpu, \
    _initialize_affine_weight_gpu, set_tensor_model_parallel_attributes, _grad_accum_fusion_available, \
    linear_with_grad_accumulation_and_async_allreduce, linear_with_frozen_weight
from megatron.core.tensor_parallel.mappings import scatter_to_tensor_model_parallel_region, \
    reduce_from_tensor_model_parallel_region, gather_from_tensor_model_parallel_region, copy_to_tensor_model_parallel_region
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region as megatron_scatter_to_sequence_parallel_region
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region as megatron_gather_from_sequence_parallel_region

from .unaligned_column_parallel_linear import UnalignedColumnParallelLinear
from .unaligned_row_parallel_linear import UnalignedRowParallelLinear
from .unaligned_utils import unaligned_divide, unaligned_scatter_to_sequence_parallel_region, \
    unaligned_reduce_scatter_to_sequence_parallel_region, unaligned_gather_from_sequence_parallel_region


class UnalignedColumnParallelLinearAdaptor(UnalignedColumnParallelLinear, ColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        config = kwargs['config']
        explicit_expert_comm = config.tensor_model_parallel_size > 1 or config.expert_model_parallel_size > 1
        if 'is_expert' not in kwargs:
            kwargs['is_expert'] = False
        if 'tp_comm_buffer_name' not in kwargs:
            kwargs['tp_comm_buffer_name'] = None

        if kwargs['is_expert'] and explicit_expert_comm and config.moe_extended_tp:
            kwargs['parallel_group'] = get_tensor_and_expert_parallel_group()
        else:
            kwargs['parallel_group'] = get_tensor_model_parallel_group()

        if kwargs['tp_comm_buffer_name'] == 'qkv':
            kwargs['fusion_number'] = (config.hidden_size + 2 * config.kv_channels * config.num_query_groups) // config.num_query_groups
        else:
            kwargs['fusion_number'] = 1

        if not config.variable_seq_lengths:
            kwargs['seq_length'] = get_args().seq_length

        kwargs['_initialize_affine_weight_cpu'] = _initialize_affine_weight_cpu
        kwargs['_initialize_affine_weight_gpu'] = _initialize_affine_weight_gpu
        kwargs['set_tensor_model_parallel_attributes'] = set_tensor_model_parallel_attributes
        kwargs['linear_with_grad_accumulation_and_async_allreduce'] = linear_with_grad_accumulation_and_async_allreduce
        kwargs['gather_from_tensor_model_parallel_region'] = gather_from_tensor_model_parallel_region
        kwargs['copy_to_tensor_model_parallel_region'] = copy_to_tensor_model_parallel_region
        kwargs['linear_with_frozen_weight'] = linear_with_frozen_weight
        super(UnalignedColumnParallelLinearAdaptor, self).__init__(*args, **kwargs)


class UnalignedRowParallelLinearAdaptor(UnalignedRowParallelLinear, RowParallelLinear):
    def __init__(self, *args, **kwargs):
        config = kwargs['config']
        explicit_expert_comm = config.tensor_model_parallel_size > 1 or config.expert_model_parallel_size > 1
        if 'is_expert' not in kwargs:
            kwargs['is_expert'] = False
        if 'tp_comm_buffer_name' not in kwargs:
            kwargs['tp_comm_buffer_name'] = None

        if kwargs['is_expert'] and explicit_expert_comm and config.moe_extended_tp:
            kwargs['parallel_group'] = get_tensor_and_expert_parallel_group()
        else:
            kwargs['parallel_group'] = get_tensor_model_parallel_group()

        if kwargs['tp_comm_buffer_name'] is not None and not kwargs['tp_comm_buffer_name'].startswith('fc'): # attention.linear_proj
            kwargs['fusion_number'] = config.hidden_size // config.num_query_groups
        else:
            kwargs['fusion_number'] = 1

        if not config.variable_seq_lengths:
            kwargs['seq_length'] = get_args().seq_length

        kwargs['_initialize_affine_weight_cpu'] = _initialize_affine_weight_cpu
        kwargs['_initialize_affine_weight_gpu'] = _initialize_affine_weight_gpu
        kwargs['linear_with_grad_accumulation_and_async_allreduce'] = linear_with_grad_accumulation_and_async_allreduce
        kwargs['scatter_to_tensor_model_parallel_region'] = scatter_to_tensor_model_parallel_region
        kwargs['linear_with_frozen_weight'] = linear_with_frozen_weight
        kwargs['reduce_from_tensor_model_parallel_region'] = reduce_from_tensor_model_parallel_region
        super(UnalignedRowParallelLinearAdaptor, self).__init__(*args, **kwargs)


def divide_adaptor(numerator, denominator):
    if numerator % denominator != 0:
        rank = torch.distributed.get_rank(group=get_tensor_model_parallel_group())
        return unaligned_divide(numerator, denominator, rank)
    return numerator // denominator


def scatter_to_sequence_parallel_region_adaptor(embeddings):
    world_size = torch.distributed.get_world_size(group=get_tensor_model_parallel_group())
    if embeddings.size()[0] % world_size != 0:
        return unaligned_scatter_to_sequence_parallel_region(embeddings, get_tensor_model_parallel_group())
    else:
        return megatron_scatter_to_sequence_parallel_region(embeddings)


def reduce_scatter_to_sequence_parallel_region_adaptor(inputs):
    group = get_tensor_model_parallel_group()
    return unaligned_reduce_scatter_to_sequence_parallel_region(inputs, group)


def gather_from_sequence_parallel_region_adaptor(inputs, tensor_parallel_output_grad=True):
    world_size = torch.distributed.get_world_size(group=get_tensor_model_parallel_group())
    
    dim_size = torch.tensor(inputs.shape[0], dtype=torch.long, device=inputs.device)
    torch.distributed.all_reduce(dim_size)
    total_dim_size = dim_size.item()
    
    group = get_tensor_model_parallel_group()
    if total_dim_size % world_size != 0:
        return unaligned_gather_from_sequence_parallel_region(inputs, group, tensor_parallel_output_grad)
    else:
        return megatron_gather_from_sequence_parallel_region(inputs, tensor_parallel_output_grad)


def get_rotary_seq_len(
    self,
    inference_params,
    transformer,
    transformer_input,
    transformer_config,
) -> float:
    if inference_params is not None:
        rotary_seq_len = inference_params.max_sequence_length
    else:
        if transformer.input_tensor is not None:
            rotary_seq_len = transformer.input_tensor.size(0)
        else:
            rotary_seq_len = transformer_input.size(0)

        if transformer_config.sequence_parallel:
            if not transformer_config.variable_seq_lengths:
                rotary_seq_len = get_args().seq_length
            else:
                rotary_seq_len = torch.Tensor([rotary_seq_len]).cuda().int()
                torch.distributed.all_reduce(rotary_seq_len, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group())
                rotary_seq_len = rotary_seq_len.item()

    rotary_seq_len *= transformer_config.context_parallel_size

    return rotary_seq_len

