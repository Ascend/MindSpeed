# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import math
from logging import getLogger
from typing import Dict, List
import torch
from megatron.core import parallel_state, tensor_parallel, mpu
from megatron.training import get_args
from megatron.core.distributed.param_and_grad_buffer import BufferType
from megatron.core.distributed.param_and_grad_buffer import shard_buffer
from megatron.training.utils import print_rank_0
from mindspeed.core.transformer.moe.moe_utils import (
    get_grouped_expert_params, 
    get_expert_param_dtype, 
    get_expert_param_size, 
    get_expert_param_data, 
    set_expert_param_data
)
from mindspeed.core.transformer.moe.expert_placement.optimizer_state_placement_memory_pool import (
    OptimizerStatePlacementMemoryPool)
from mindspeed.core.transformer.moe.expert_placement.planner import gather_expert_load_data_parallel



def build_param_params_module_mlp_map(model_chunks):
    params_module_mlp_map = {}
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if (".mlp.experts.weight" in name or ".mlp.router" in name) and "mtp.layers" not in name:
                parts = name.split('layers.')
                layer_number = int(parts[1].split('.')[0])
                module_mlp = model_chunk.module.module.decoder.layers[layer_number].mlp
                params_module_mlp_map.update({param: [module_mlp, name]})
            elif (".mlp.experts.weight" in name or ".mlp.router" in name) and "mtp.layers" in name:
                parts = name.split('layers.')
                layer_number = int(parts[1].split('.')[0])
                module_mlp = model_chunk.module.module.mtp.layers[layer_number].transformer_layer.mlp
                params_module_mlp_map.update({param: [module_mlp, name]})
    return params_module_mlp_map


def expert_weight_and_optimizer_state_placement(args, model, optimizer):
    # relocate model parameters(experts) and corresponding optimizer states 
    expert_placement_triggered = args.curr_iteration % args.expert_placement_freq == args.expert_placement_freq - 1 
    if expert_placement_triggered:
        
        for model_chunk in model:
            for layer in model_chunk.module.module.decoder.layers:
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router'):
                    initialize_expert_placement_dispatcher(layer.mlp)
                    if layer.mlp.expert_placement_layer_permission:
                        expert_weight_placement(layer.mlp)
                        router_weight_placement(layer.mlp)
                        if args.moe_router_enable_expert_bias:
                            expert_bias_and_local_token_count_placement(layer.mlp)

            if hasattr(model_chunk.module.module, 'mtp'):
                for layer in model_chunk.module.module.mtp.layers:
                    if (
                        hasattr(layer, 'transformer_layer') 
                        and hasattr(layer.transformer_layer, 'mlp') 
                        and hasattr(layer.transformer_layer.mlp, 'router')
                    ):  
                        initialize_expert_placement_dispatcher(layer.transformer_layer.mlp)
                        if layer.transformer_layer.mlp.expert_placement_layer_permission:
                            expert_weight_placement(layer.transformer_layer.mlp)
                            router_weight_placement(layer.transformer_layer.mlp)
                            if args.moe_router_enable_expert_bias:
                                expert_bias_and_local_token_count_placement(layer.transformer_layer.mlp)
        if hasattr(optimizer, "chained_optimizers"):
            for sub_optimizer in optimizer.chained_optimizers:
                relocate_expert_optimization_states(sub_optimizer, reuse_fp32_param=args.reuse_fp32_param)
        else:
            relocate_expert_optimization_states(optimizer, reuse_fp32_param=args.reuse_fp32_param)


def initialize_expert_placement_dispatcher(mlp):
    with torch.no_grad():
        mlp.expert_load_prediction.copy_(gather_expert_load_data_parallel(mlp.expert_load_prediction))
        # Calc the KL divergence between current and former expert load
        # Calc expert placement mapping
        ep_devices = list(range(mlp.ep_world_size))
        expert_load = dict(zip(range(mlp.config.num_moe_experts), 
                               mlp.expert_load_prediction.to(torch.device("cpu")).numpy()))
        expert_device_mapping, device_expert_mapping = \
            mlp.expert_placement_optimizer.expert_placement_greedy(expert_load, ep_devices)
        
        # Turn expert_device_mapping into expert_index mapping
        device_expert_idx = [set() for i in range(mlp.ep_world_size)]
        device_full_expert_idx = {i for i in range(mlp.experts.num_local_experts)}

        new_expert_mapping = mlp.expert_mapping.clone()
        for i in range(mlp.config.num_moe_experts):
            if mlp.expert_mapping[i] // mlp.experts.num_local_experts == expert_device_mapping[i]:
                local_idx = mlp.expert_mapping[i] % mlp.experts.num_local_experts
                device_expert_idx[expert_device_mapping[i][0]].append(local_idx)
                continue
            remaining_indices = device_full_expert_idx - device_expert_idx[expert_device_mapping[i][0]]
            local_idx = remaining_indices.pop()
            new_expert_mapping[i] = expert_device_mapping[i][0] * mlp.experts.num_local_experts + local_idx
            device_expert_idx[expert_device_mapping[i][0]].add(local_idx)

        # Calc expert load variation
        args = get_args()
        
        mlp.expert_placement_layer_permission = True
        if args.enable_fine_grained_expert_placement:
            sorted_expert_mapping, sorted_expert_mapping_indices = torch.sort(mlp.expert_mapping)
            expert_load_per_device_pre = mlp.expert_load_prediction[sorted_expert_mapping_indices].reshape(
                                                                            -1, mlp.experts.num_local_experts).sum(-1)
            expert_load_variation_pre = expert_load_per_device_pre.std() / expert_load_per_device_pre.mean() \
                                        if expert_load_per_device_pre.mean() > 0 else expert_load_per_device_pre.std()

            sorted_expert_mapping, sorted_expert_mapping_indices = torch.sort(new_expert_mapping)
            expert_load_per_device_curr = mlp.expert_load_prediction[sorted_expert_mapping_indices].reshape(
                                                                            -1, mlp.experts.num_local_experts).sum(-1)
            expert_load_variation_curr = expert_load_per_device_curr.std() / expert_load_per_device_curr.mean() \
                                        if expert_load_per_device_curr.mean() > 0 else expert_load_per_device_curr.std()

            expert_load_variation_diff = expert_load_variation_pre - expert_load_variation_curr
            expert_load_variation_diff_thre = args.fine_grained_expert_placement_thre
            if expert_load_variation_diff < expert_load_variation_diff_thre:
                mlp.expert_placement_layer_permission = False
            if torch.distributed.get_rank() == 0:
                print(f"expert_load_variation_pre {expert_load_variation_pre} \
                        expert_load_variation_curr {expert_load_variation_curr}")

        if not mlp.expert_placement_layer_permission:
            return

        # permuting the experts according to expert_device_mapping        
        if args.moe_tp_extend_ep:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            local_expert_indices_offset = (
                    parallel_state.get_expert_model_parallel_rank() * mlp.experts.num_local_experts * tp_size + \
                    parallel_state.get_tensor_model_parallel_rank() * mlp.experts.num_local_experts
            )
        else:
            local_expert_indices_offset = (
                parallel_state.get_expert_model_parallel_rank() * mlp.experts.num_local_experts
            )

        experts_local_indices_mask = torch.logical_and(local_expert_indices_offset <= mlp.expert_mapping,
                                                       mlp.expert_mapping < local_expert_indices_offset +
                                                       mlp.experts.num_local_experts)
        old_local_expert_mapping = mlp.expert_mapping[experts_local_indices_mask].clone()
        new_local_expert_mapping = new_expert_mapping[experts_local_indices_mask].clone()

        sorted_old_local_expert_mapping, old_local_expert_sorted_indices = torch.sort(old_local_expert_mapping)
        new_local_expert_mapping = new_local_expert_mapping[old_local_expert_sorted_indices]
        sorted_local_expert_mapping, new_local_expert_sorted_indices = torch.sort(new_local_expert_mapping)
        sorted_local_expert_mapping = sorted_local_expert_mapping // mlp.experts.num_local_experts
        mlp.input_splits_for_expert_placement = torch.histc(sorted_local_expert_mapping, 
                                                             bins=mlp.ep_world_size, 
                                                             min=0, 
                                                             max=mlp.ep_world_size).cpu().numpy()

        sorted_expert_mapping, sorted_expert_mapping_indices = torch.sort(mlp.expert_mapping)
        relative_mapping = new_expert_mapping[sorted_expert_mapping_indices].reshape(mlp.ep_world_size, -1)
        sorted_relative_mapping, sorted_relative_mapping_indices = torch.sort(relative_mapping, dim=1)
        new_local_device_mapping_mask = (sorted_relative_mapping >= local_expert_indices_offset) & \
                                (sorted_relative_mapping < local_expert_indices_offset + mlp.experts.num_local_experts)
        local_sorted_relative_mapping = sorted_relative_mapping[new_local_device_mapping_mask]
        _, local_resorted_relative_mapping_indices = torch.sort(local_sorted_relative_mapping)
        mlp.output_splits_for_expert_placement = new_local_device_mapping_mask.sum(axis=1).cpu().numpy()

        mlp.expert_mapping.copy_(new_expert_mapping)
        mlp.relative_mapping = relative_mapping.view(-1)
        mlp.local_resorted_relative_mapping_indices = local_resorted_relative_mapping_indices
        mlp.new_local_expert_sorted_indices.copy_(new_local_expert_sorted_indices)
 

def expert_weight_placement(mlp):
    global_args = get_args()
    print_rank_0(f"[expert placement] iter {global_args.curr_iteration}: expert_placement_triggered!!!")
    with torch.no_grad():
        if mlp.config.moe_extended_tp:
            return
        if not mlp.config.moe_tp_extend_ep:
            tp_size = 1
            expert_placement_group = parallel_state.get_expert_model_parallel_group()
        else:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            expert_placement_group = parallel_state.get_expert_model_parallel_group()

        # determine mlp type, todo
        group_mlp_expert_params = None
        if mlp.config.moe_grouped_gemm:
            group_mlp_expert_params = get_grouped_expert_params(mlp.experts, 
                                                                mlp.experts.num_local_experts, 
                                                                tp_size, mlp.experts.config)
        params_size = get_expert_param_size(mlp.experts, group_mlp_expert_params, 0)
        params_type = get_expert_param_dtype(mlp.experts, group_mlp_expert_params, 0)

        input_paras_buffer = torch.zeros((mlp.experts.num_local_experts, params_size), 
                                         dtype=params_type, device=torch.cuda.current_device())

        for i in range(mlp.experts.num_local_experts):
            get_expert_param_data(group_mlp_expert_params, input_paras_buffer[i], 
                                  mlp.new_local_expert_sorted_indices[i].item())
        
        #all2all communication and set paras     
        output_paras_buffer = tensor_parallel.all_to_all(
            expert_placement_group,
            input_paras_buffer,
            mlp.output_splits_for_expert_placement,
            mlp.input_splits_for_expert_placement,
        )
        for idx in range(mlp.experts.num_local_experts):
            set_expert_param_data(group_mlp_expert_params, 
                                  output_paras_buffer[mlp.local_resorted_relative_mapping_indices[idx].item()], idx)


def router_weight_placement(mlp):
    inverse_router_mapping = torch.argsort(mlp.relative_mapping)
    with torch.no_grad():
        reordered_data = mlp.router.weight.data[inverse_router_mapping]
        mlp.router.weight.data.copy_(reordered_data)


def expert_bias_and_local_token_count_placement(mlp):
    inverse_router_mapping = torch.argsort(mlp.relative_mapping)
    with torch.no_grad():
        reordered_bias_data = mlp.router.expert_bias[inverse_router_mapping]
        reordered_tokens_data = mlp.router.local_tokens_per_expert[inverse_router_mapping]
        mlp.router.expert_bias.data.copy_(reordered_bias_data)
        mlp.router.local_tokens_per_expert.data.copy_(reordered_tokens_data)


def get_bucket_param_placement_permission(optimizer, bucket):
    bucket_param_placement_permission = False
    for param in bucket.params_list:
        if param in optimizer.params_module_mlp_map.keys() and \
                                            optimizer.params_module_mlp_map[param][0].expert_placement_layer_permission:
            bucket_param_placement_permission = True
            break
    return bucket_param_placement_permission


def relocate_expert_optimization_states(optimizer, reuse_fp32_param=True):
    '''
    placement the states of optimizer provided the conditions are satified
    ''' 

    pool = OptimizerStatePlacementMemoryPool()
    for buffer in optimizer.buffers:
        for _, bucket in enumerate(buffer.buckets):
            bucket_param_placement_permission = get_bucket_param_placement_permission(optimizer, bucket)

            if not bucket_param_placement_permission:
                continue
            if reuse_fp32_param:
                dtype = torch.bfloat16
                optimizer_state_placement_memory = pool.get((bucket.param_data.nelement(),), dtype)
                # relocate optimizer res 
                bidirectional_copy_bucket_optimizer_buffer_with_optimizer_res(
                                        optimizer, bucket, optimizer_state_placement_memory, bucket_buffer_to_res=False)
                optbuf_view_items_dp = _get_optimizer_state_buffer_dp_views(optimizer, optimizer_state_placement_memory)
                all_gather_handle = _dispatch_gather_optimizer_states(optimizer, optbuf_view_items_dp, force_sync=False)
                optbuf_view_items_ep = _get_optimizer_state_buffer_ep_views(
                                                            optimizer, buffer, bucket, optimizer_state_placement_memory)
                _dispatch_all2all_optimizer_states(optimizer, optbuf_view_items_ep)
                bidirectional_copy_bucket_optimizer_buffer_with_optimizer_res(
                                        optimizer, bucket, optimizer_state_placement_memory, bucket_buffer_to_res=True)

            else:
                dtype = torch.float32
                optimizer_state_placement_memory = pool.get((bucket.param_data.nelement(),), dtype)
                bidirectional_copy_bucket_optimizer_buffer_with_optimizer_mainparam(
                                optimizer, bucket, optimizer_state_placement_memory, bucket_buffer_to_mainparam=False)
                optbuf_view_items_dp = _get_optimizer_state_buffer_dp_views(optimizer, optimizer_state_placement_memory)
                all_gather_handle = _dispatch_gather_optimizer_states(optimizer, optbuf_view_items_dp, force_sync=False)
                optbuf_view_items_ep = _get_optimizer_state_buffer_ep_views(
                                optimizer, buffer, bucket, optimizer_state_placement_memory)
                _dispatch_all2all_optimizer_states(optimizer, optbuf_view_items_ep)
                bidirectional_copy_bucket_optimizer_buffer_with_optimizer_mainparam(
                                optimizer, bucket, optimizer_state_placement_memory, bucket_buffer_to_mainparam=True)

            # relocate optimizer states
            for key in ["exp_avg", "exp_avg_sq"]:
                dtype = torch.float32
                optimizer_state_placement_memory = pool.get((bucket.param_data.nelement(),), dtype)
                bidirectional_copy_bucket_optimizer_buffer_with_optimizer_states(
                                optimizer, bucket, optimizer_state_placement_memory, key, bucket_buffer_to_state=False)
                optbuf_view_items_dp = _get_optimizer_state_buffer_dp_views(optimizer, optimizer_state_placement_memory)
                all_gather_handle = _dispatch_gather_optimizer_states(optimizer, optbuf_view_items_dp, force_sync=False)
                optbuf_view_items_ep = _get_optimizer_state_buffer_ep_views(
                                optimizer, buffer, bucket, optimizer_state_placement_memory)
                _dispatch_all2all_optimizer_states(optimizer, optbuf_view_items_ep)
                bidirectional_copy_bucket_optimizer_buffer_with_optimizer_states(
                                optimizer, bucket, optimizer_state_placement_memory, key, bucket_buffer_to_state=True)
            
            pool.clear()

 
def bidirectional_copy_bucket_optimizer_buffer_with_optimizer_mainparam(
    optimizer, 
    bucket, 
    optimizer_state_placement_memory, 
    bucket_buffer_to_mainparam=True):
    """
    Copy main params to model params.
 
    Since this step is followed by an all-gather through the DDP's grad
    buffer, this method is responsible for copying the updated params
    from the main shards into the correct position in the grad buffer.
    """
    def copy_group_params(shard_main_groups, model_groups, bucket_buffer_to_mainparam=True):
        for shard_main_group, model_group in zip(shard_main_groups, model_groups):
            for shard_main_param, model_param in zip(shard_main_group, model_group):
                if not (
                    model_param in optimizer.params_module_mlp_map
                    and optimizer.params_module_mlp_map[model_param][0].expert_placement_layer_permission 
                    and model_param in bucket.params
                ):
                    continue

                param_range_map = optimizer._get_model_param_range_map(model_param)
                world_range = param_range_map["gbuf_world_in_bucket"]
 
                shard_bucket_optimizer_state_buffer = (
                    optimizer_state_placement_memory.view(-1)[world_range.start:world_range.end]
                )
                if (
                    getattr(get_args(), "swap_optimizer", False) 
                    and shard_main_param in optimizer.param_to_cpu_states_map
                ):
                    candidate_main_param = optimizer.param_to_cpu_states_map[shard_main_param]['param']
                else:
                    candidate_main_param = shard_main_param
                if bucket_buffer_to_mainparam:
                    candidate_main_param.copy_(shard_bucket_optimizer_state_buffer)
                else:
                    shard_bucket_optimizer_state_buffer.copy_(candidate_main_param)

    # Copy optimizer mainparams to buffer.
    copy_group_params(optimizer.shard_fp32_from_float16_groups,
                      optimizer.model_float16_groups,
                      bucket_buffer_to_mainparam)
    copy_group_params(optimizer.shard_fp32_groups, optimizer.model_fp32_groups, bucket_buffer_to_mainparam)
 
 
def bidirectional_copy_bucket_optimizer_buffer_with_optimizer_res(
    optimizer, 
    bucket, 
    optimizer_state_placement_memory, 
    bucket_buffer_to_res=True):
    # Utility method for copying group params.
    for model_group in optimizer.model_float16_groups:
        for model_param in model_group:
            if not (
                model_param in optimizer.params_module_mlp_map
                and optimizer.params_module_mlp_map[model_param][0].expert_placement_layer_permission 
                and model_param in bucket.params
            ):
                continue

            param_range_map = optimizer._get_model_param_range_map(model_param)
            world_range = param_range_map["gbuf_world_in_bucket"]
            param_local_range = param_range_map["gbuf_local"]
 
            gbuf_index, _, bucket_id = optimizer.model_param_gbuf_map[model_param]
            data_parallel_world_size = torch.distributed.get_world_size(optimizer.data_parallel_group)
            if data_parallel_world_size == 1:
                data_start_index, data_end_index, bucket_id = optimizer.buffers[gbuf_index].param_index_map[model_param]
                shard_optimizer_res = optimizer.shard_main_param_res_buffers[gbuf_index][
                    2 * data_start_index:data_start_index + data_end_index
                ]
            else:
                bucket_param_data = optimizer.buffers[gbuf_index].buckets[bucket_id].param_data
                bucket_res = optimizer.model_param_bucket_and_res_map[bucket_param_data]
                shard_optimizer_res = bucket_res[param_local_range.start: param_local_range.end]

            shard_bucket_optimizer_state_buffer = (
                optimizer_state_placement_memory.view(-1)[world_range.start:world_range.end]
            )

            if bucket_buffer_to_res:
                shard_optimizer_res.copy_(shard_bucket_optimizer_state_buffer)
            else:
                shard_bucket_optimizer_state_buffer.copy_(shard_optimizer_res)

 
def bidirectional_copy_bucket_optimizer_buffer_with_optimizer_states(
    optimizer, 
    bucket, 
    optimizer_state_placement_memory, 
    key, 
    bucket_buffer_to_state=True):
    # Utility method for copying group params.
    def copy_group_params(model_groups, key, bucket_buffer_to_state=True):
        for model_group in model_groups:
            for model_param in model_group:
                if not (
                    model_param in optimizer.params_module_mlp_map.keys() 
                    and optimizer.params_module_mlp_map[model_param][0].expert_placement_layer_permission 
                    and model_param in bucket.params
                ):
                    continue
                
                param_range_map = optimizer._get_model_param_range_map(model_param)
                world_range = param_range_map["gbuf_world_in_bucket"]
  
                shard_bucket_optimizer_state_buffer = optimizer_state_placement_memory.view(-1)[
                    world_range.start:world_range.end
                ]

                group_index, group_order = optimizer.model_param_group_index_map[model_param]
                main_param = optimizer.optimizer.param_groups[group_index]["params"][group_order]
                if key in optimizer.optimizer.state[main_param].keys():
                    if getattr(get_args(), "swap_optimizer", False) and main_param in optimizer.param_to_cpu_states_map:
                        candidate_optimizer_state = optimizer.param_to_cpu_states_map[main_param][key]
                    else:
                        candidate_optimizer_state = optimizer.optimizer.state[main_param][key]

                    if bucket_buffer_to_state:
                        candidate_optimizer_state.copy_(shard_bucket_optimizer_state_buffer)
                    else:
                        shard_bucket_optimizer_state_buffer.copy_(candidate_optimizer_state)

    copy_group_params(optimizer.model_float16_groups, key, bucket_buffer_to_state)
    copy_group_params(optimizer.model_fp32_groups, key, bucket_buffer_to_state)
 
 
def _get_optimizer_state_buffer_dp_views(optimizer, optimizer_state_placement_memory):
    """
    Get shard views of moe optimizer state buffers of float dtype.
 
    In this nested list, the top level is grouped by the virtual model
    index and the buffer's data type. The sub-level is a list of
    shards of that buffer, where each shard in the list represents
    a contiguous view of the buffer, that is owned by a data-parallel
    rank. The shard boundary does not respect parameter boundaries, and
    so the elements of some parameters are split across data parallel
    ranks.
 
    Additionally, return references to the entire buffers, for use
    in _all_gather_base.
    """
 
    # Buffer views.
    # Add in reverse order in each model chunk since buckets start from the end of the model but we want
    # all-gathers to run first for the start of the model (same order as forward pass).
    # We keep the view_items in model chunk order since we want to still first run all_gather and
    # all_gather_handle.wait() for the first model chunk.
    # In all cases, we want all_gather and all_gather_handle.wait() to be called in the same order,
    # and all_gather_handle.wait() needs to be called just before the corresponding forward pass.
    view_items = []
    dtype = 'torch.cuda.FloatTensor'
    data_parallel_world_size = torch.distributed.get_world_size(
        optimizer.data_parallel_group
    )
    buf_views = shard_buffer(optimizer_state_placement_memory, data_parallel_world_size)

    view_items.insert(0, (dtype, optimizer_state_placement_memory, buf_views))
    return view_items
 
 
def _dispatch_gather_optimizer_states(optimizer, optbuf_view_items_dp, force_sync: bool = False):
    """
    All-gather updated optimizer states.
 
    When using the distributed optimizer, the states are already laid out in a contiguous
    buffer (see mcore/distributed/param_and_grad_buffer.py for details), and so the
    all-gather will put the results in the right region of memory.
    """
    async_op = optimizer.ddp_config.overlap_param_gather and not force_sync
    data_parallel_group = optimizer.data_parallel_group
    data_parallel_rank = torch.distributed.get_rank(data_parallel_group)

    # All paramoptbuf_buf views are guaranteed to have the same number of elements
    # across all data-parallel ranks, due to padding done in
    # param_and_grad_buffer.py). Thus, all sub-views will have consistent
    # start / end indexes across data-parallel ranks.
    (dtype, optbuf, optbuf_views) = optbuf_view_items_dp[0]
    if optbuf is not None:
        all_gather_handle = torch.distributed._all_gather_base(
            optbuf, optbuf_views[data_parallel_rank], group=data_parallel_group, async_op=async_op,
        )
    else:
        all_gather_handle = None
    return all_gather_handle
  
 
def _get_optimizer_state_buffer_ep_views(optimizer, buffer, bucket, optimizer_state_placement_memory):
    """
    Get shard views of moe optimizer state buffers of float dtype.
 
    In this nested list, the top level is grouped by the virtual model
    index and the buffer's data type. The sub-level is a list of
    shards of that buffer, where each shard in the list represents
    a contiguous view of the buffer, that is owned by a data-parallel
    rank. The shard boundary does not respect parameter boundaries, and
    so the elements of some parameters are split across data parallel
    ranks.
 
    Additionally, return references to the entire buffers, for use
    in _all_gather_base.
    """
 
    view_items = []
    for _, param in enumerate(bucket.params_list):
        if not (
            param in optimizer.params_module_mlp_map
            and optimizer.params_module_mlp_map[param][0].expert_placement_layer_permission 
            and param in bucket.params
        ):
            continue

        param_world_start_index, param_world_end_index, _ = buffer.param_index_map[param]
        param_bucket_start_index = param_world_start_index - bucket.offset
        param_bucket_end_index = param_world_end_index - bucket.offset
        param_optimizer_data = optimizer_state_placement_memory[param_bucket_start_index:param_bucket_end_index]

        if ".mlp.router" in optimizer.params_module_mlp_map[param][1]:
            param_optimizer_data_reshape = param_optimizer_data.view(
                torch.Size([optimizer.params_module_mlp_map[param][0].config.num_moe_experts, -1])
            )
            inverse_expert_mapping = torch.argsort(optimizer.params_module_mlp_map[param][0].relative_mapping)
            param_optimizer_data_reshape.copy_(param_optimizer_data_reshape[inverse_expert_mapping])
        else:
            input_splits = None
            output_splits = None
            local_resorted_relative_mapping_indices = None
            new_local_expert_sorted_indices = None
            input_splits = optimizer.params_module_mlp_map[param][0].input_splits_for_expert_placement
            output_splits = optimizer.params_module_mlp_map[param][0].output_splits_for_expert_placement
            local_resorted_relative_mapping_indices = (
                optimizer.params_module_mlp_map[param][0].local_resorted_relative_mapping_indices
            )
            new_local_expert_sorted_indices = optimizer.params_module_mlp_map[param][0].new_local_expert_sorted_indices
            
            num_local_experts = optimizer.params_module_mlp_map[param][0].experts.num_local_experts
            numele_per_expert = int((param_bucket_end_index - param_bucket_start_index) / num_local_experts)

            param_optimizer_data_reshape = param_optimizer_data.view(
                torch.Size([num_local_experts, numele_per_expert])
            )

            view_items.insert(
                0, 
                (param_optimizer_data_reshape,
                 input_splits, 
                 output_splits, 
                 local_resorted_relative_mapping_indices, 
                 new_local_expert_sorted_indices, 
                 optimizer.params_module_mlp_map[param][1]
                )
            )
    view_items_sorted = sorted(view_items, key=lambda x: x[5])
    return view_items_sorted
 
 
def _dispatch_all2all_optimizer_states(optimizer, optbuf_view_items_ep):
    """
    all2all updated optimizer states.
 
    When using the distributed optimizer, the states are already laid out in a contiguous
    buffer (see mcore/distributed/param_and_grad_buffer.py for details), and so the
    all2all will put the results in the right region of memory.
    """
    for optbuf_view_item in optbuf_view_items_ep:
        (
            param_optimizer_data_reshape, 
            input_splits, output_splits, 
            local_resorted_relative_mapping_indices, 
            new_local_expert_sorted_indices, 
            param_name
        ) = optbuf_view_item
        if param_optimizer_data_reshape is not None:
            global_args = get_args()
            if global_args.moe_tp_extend_ep:
                optimizer_state_placement_group = parallel_state.get_tensor_and_expert_parallel_group()
            else:
                optimizer_state_placement_group = parallel_state.get_expert_model_parallel_group()
 
            param_optimizer_data_relocate = tensor_parallel.all_to_all(
                optimizer_state_placement_group,
                param_optimizer_data_reshape[new_local_expert_sorted_indices], 
                output_splits, 
                input_splits
            )
            param_optimizer_data_reshape.copy_(param_optimizer_data_relocate[local_resorted_relative_mapping_indices])

