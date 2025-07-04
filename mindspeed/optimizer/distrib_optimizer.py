# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Bytedance Inc. All rights reserved.
import os
from textwrap import wrap
import types
from functools import wraps
import torch
import torch_npu
import torch.distributed
import megatron.core.tensor_parallel as tensor_parallel

from megatron.training import get_args

TRANSPOSE_BF16_BLOCK_SIZE = 4096 * 4096


def build_optimizer_group_ranges_zero3_wrapper(fn):
    @wraps(fn)
    def build_optimizer_group_ranges_zero3(param_groups, gbuf_ranges):
        local_param_group_map, group_ranges = fn(param_groups, gbuf_ranges)
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                if hasattr(param, 'enable_zero3') and param.enable_zero3:
                    group_range = group_ranges[group_index]
                    group_range["params"].append(param)
                    local_param_group_map[param] = (group_index, len(group_range["params"]) - 1)
        return local_param_group_map, group_ranges
    return build_optimizer_group_ranges_zero3


def _copy_main_params_to_model_params_zero3(self):
    """
    Copy main params to model params.

    Since this step is followed by an all-gather through the DDP's grad
    buffer, this method is responsible for copying the updated params
    from the main shards into the correct position in the grad buffer.
    """
    # Utility method for copying group params.
    def copy_group_params(shard_main_groups, model_groups):
        for shard_main_group, model_group in zip(shard_main_groups, model_groups):
            for shard_main_param, model_param in zip(shard_main_group, model_group):
                if hasattr(model_param, 'enable_zero3') and model_param.enable_zero3:
                    model_param.view(-1).data.copy_(shard_main_param)
                    continue
                param_range_map = self._get_model_param_range_map(model_param)
                world_range = param_range_map["gbuf_world_in_bucket"]

                assert world_range.size == shard_main_param.nelement()

                gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                model_param_buffer = self.buffers[gbuf_index].buckets[bucket_id].param_data

                shard_model_param = model_param_buffer.view(-1)[
                    world_range.start : world_range.end
                ]

                shard_model_param.data.copy_(shard_main_param)

    # Copy shard groups to model groups.
    copy_group_params(self.shard_fp32_from_float16_groups, self.model_float16_groups)
    copy_group_params(self.shard_fp32_groups, self.model_fp32_groups)


def build_model_and_main_param_groups_zero3_wrapper(function):
    @wraps(function)
    def build_model_and_main_param_zero3(*args, **kwargs):
        global_args = get_args()
        if global_args.enable_zero3:
            return build_model_and_main_param_groups_zero3(*args, **kwargs)
        else:
            return function(*args, **kwargs)
    return build_model_and_main_param_zero3


def build_model_and_main_param_groups_zero3(
    gbuf_ranges,
    param_gbuf_map,
    opt_group_ranges,
):
    """
    Create main parameter groups needed for the optimizer step.

    These groups encompass both: 1) groups used by this class, for
    reducing/gather, and 2) groups used by the inner optimizer for the
    parameter update. Given that the conceptual grad buffer partitioning
    (created in earlier method) doesn't respect parameter boundaries,
    the optimizer operates on shards of the model parameters, rather than
    the full parameters.
    """

    # Parameter groups:
    #   model_float16_groups: original float16 parameters
    #   model_fp32_groups: original fp32 parameters
    #   shard_float16_groups: shards of original float16 parameters
    #   shard_fp32_groups: shards of original fp32 parameters
    #   shard_fp32_from_float16_groups: fp32 copy of float16 parameters
    model_float16_groups = []
    model_fp32_groups = []
    shard_float16_groups = []
    shard_fp32_groups = []
    shard_fp32_from_float16_groups = []

    # Allocate (or slice) each group's param shard.
    for group_range in opt_group_ranges:

        # Params of this group.
        model_float16_params_this_group = []
        model_fp32_params_this_group = []
        shard_float16_params_this_group = []
        shard_fp32_params_this_group = []
        shard_fp32_from_float16_params_this_group = []
        model_float16_groups.append(model_float16_params_this_group)
        model_fp32_groups.append(model_fp32_params_this_group)
        shard_float16_groups.append(shard_float16_params_this_group)
        shard_fp32_groups.append(shard_fp32_params_this_group)
        shard_fp32_from_float16_groups.append(shard_fp32_from_float16_params_this_group)

        for model_param in group_range["params"]:

            assert model_param.requires_grad
            
            if hasattr(model_param, 'enable_zero3') and model_param.enable_zero3:
                if model_param.type() in ['torch.cuda.HalfTensor',
                                            'torch.cuda.BFloat16Tensor',
                                            'torch.BFloat16Tensor']:
                    zero3_main_param = model_param.detach().view(-1).clone().float()
                    tensor_parallel.copy_tensor_model_parallel_attributes(zero3_main_param, model_param)
                    model_float16_params_this_group.append(model_param)
                    shard_float16_params_this_group.append(model_param)
                    shard_fp32_from_float16_params_this_group.append(zero3_main_param)
                elif model_param.type() == 'torch.cuda.FloatTensor':
                    model_fp32_params_this_group.append(model_param)
                    zero3_fp32_main_param = model_param.view(-1)
                    shard_fp32_params_this_group.append(zero3_fp32_main_param)
                    tensor_parallel.copy_tensor_model_parallel_attributes(zero3_fp32_main_param, model_param)
                else:
                    raise TypeError(
                    'Wrapped parameters must be one of '
                    'torch.cuda.FloatTensor,  '
                    'torch.cuda.HalfTensor, or '
                    'torch.cuda.BFloat16Tensor. '
                    'Received {}'.format(model_param.type())
                )
                continue


            gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
            gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
            param_range = gbuf_range["param_map"][model_param]["param"]

            # fp16, bf16 params.
            if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:

                # Clone model -> main.
                shard_model_param = model_param.detach().view(-1)[
                    param_range.start : param_range.end
                ]
                shard_main_param = shard_model_param.clone().float()
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_model_param, model_param
                )
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_main_param, model_param
                )
                if hasattr(model_param, 'shared'):
                    shard_model_param.shared = model_param.shared
                    shard_main_param.shared = model_param.shared

                # Add to group.
                model_float16_params_this_group.append(model_param)
                shard_float16_params_this_group.append(shard_model_param)
                shard_fp32_from_float16_params_this_group.append(shard_main_param)

            # fp32 params.
            elif model_param.type() == 'torch.cuda.FloatTensor':
                shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
                model_fp32_params_this_group.append(model_param)
                shard_fp32_params_this_group.append(shard_model_param)
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_model_param, model_param
                )
                if hasattr(model_param, 'shared'):
                    shard_model_param.shared = model_param.shared

            else:
                raise TypeError(
                    'Wrapped parameters must be one of '
                    'torch.cuda.FloatTensor,  '
                    'torch.cuda.HalfTensor, or '
                    'torch.cuda.BFloat16Tensor. '
                    'Received {}'.format(model_param.type())
                )

        # Update optimizer's params.
        group_range["orig_group"]["params"] = [
            *shard_fp32_params_this_group,
            *shard_fp32_from_float16_params_this_group,
        ]

    return (
        model_float16_groups,
        model_fp32_groups,
        shard_float16_groups,
        shard_fp32_groups,
        shard_fp32_from_float16_groups,
    )


def distributed_optimizer_zero3_init(
    self,
    optimizer,
    config,
    grad_scaler,
    init_state_fn,
    per_model_buffers,
    data_parallel_group,
    data_parallel_group_gloo,
    data_parallel_group_idx,
):
    """
    Distributed optimizer, for all data types (fp16, bf16, and fp32).

    The steps in this method create the core mapping between param and grad buffers,
    parameters, and parameter shard ranges, that is needed for converting between model
    param indexes and main parameter shard indexes. This method also updates the optimizer
    parameter groups with the newly created shards.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        grad_scaler (MegatronGradScaler): used for scaling gradients. Note that
            this can be None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constant gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
        per_model_buffers (Dict[int, List[ParamAndGradBuffer]]): the implementation of the
            distributed optimizer is centered on using a contiguous buffer for
            communicating grads & params between the model state and the optimizer state.
            You can find a more detailed description in
            https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md.
        data_parallel_group (torch.distributed.ProcessGroup): data-parallel group to use to
            all-gather params after optimizer.step().
        data_parallel_group_gloo (torch.distributed.ProcessGroup): gloo data-parallel group
            (used in checkpoint loading and saving).
        data_parallel_group_idx (int): index in data-parallel group (used by
            distributed checkpointing logic).
    """

    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from apex.optimizers import FusedAdam as Adam
    import itertools

    super(self.__class__, self).__init__(
        optimizer, config, grad_scaler, init_state_fn,
    )

    assert isinstance(
        optimizer, Adam
    ), "Only Adam currently supported, due to checkpointing requirements."

    # Model grad buffer ranges.
    assert per_model_buffers is not None, "per_model_buffers must be provided"
    self.buffers = list(itertools.chain(*per_model_buffers.values()))
    self.per_model_buffers = per_model_buffers
    self.data_parallel_group = data_parallel_group
    self.data_parallel_group_gloo = data_parallel_group_gloo
    self.data_parallel_group_idx = data_parallel_group_idx
    self.gbuf_idx_to_model_idx_map = {}
    gbuf_idx = 0
    for model_idx, buffers in self.per_model_buffers.items():
        for _ in buffers:
            self.gbuf_idx_to_model_idx_map[gbuf_idx] = model_idx
            gbuf_idx += 1
    self.gbuf_ranges = []
    self.per_bucket_numel = []
    self.per_bucket_numel_unpadded = []
    for buffer in self.buffers:

        self.per_bucket_numel.append(
            {
                (buffer.param_dtype, buffer.grad_dtype): [
                    bucket.grad_data.numel() for bucket in buffer.buckets
                ]
            }
        )
        self.per_bucket_numel_unpadded.append(
            {
                (buffer.param_dtype, buffer.grad_dtype): [
                    bucket.numel_unpadded for bucket in buffer.buckets
                ]
            }
        )
        self.gbuf_ranges.append(DistributedOptimizer._build_gbuf_range_map(buffer))
    self.model_param_gbuf_map = DistributedOptimizer._build_model_param_gbuf_map(self.gbuf_ranges)

    # Optimizer ranges.
    (
        self.model_param_group_index_map,
        self.opt_group_ranges,
    ) = DistributedOptimizer._build_optimizer_group_ranges(self.optimizer.param_groups, self.gbuf_ranges)

    # Allocate main param shards.
    (
        self.model_float16_groups,
        self.model_fp32_groups,
        self.shard_float16_groups,
        self.shard_fp32_groups,
        self.shard_fp32_from_float16_groups,
    ) = DistributedOptimizer._build_model_and_main_param_groups(
        self.gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges
    )

    # Now construct data structures to manage all-gather handles.
    self.all_gather_handles = []
    self.all_gather_handle_index_to_bucket_index_map = []
    self.model_index_to_all_gather_handle_index_map = {}
    self.all_gather_handle_indices = []
    self.param_to_all_gather_handle_index_map = {}

    self.pbuf_view_items = self._get_model_param_buffer_dp_views()
    for (gbuf_index, dtype, bucket_index, _, _) in self.pbuf_view_items:
        self.all_gather_handle_index_to_bucket_index_map.append(
            (gbuf_index, dtype, bucket_index)
        )
        all_gather_handle_index = len(self.all_gather_handle_index_to_bucket_index_map) - 1
        self.all_gather_handles.append(None)

        # Store all all_gather_handle_indices.
        model_idx = self.gbuf_idx_to_model_idx_map[gbuf_index]
        if model_idx not in self.model_index_to_all_gather_handle_index_map:
            self.model_index_to_all_gather_handle_index_map[model_idx] = []
        self.model_index_to_all_gather_handle_index_map[model_idx].append(
            all_gather_handle_index
        )

        for param in self.buffers[gbuf_index].buckets[bucket_index].params_list:
            self.param_to_all_gather_handle_index_map[param] = all_gather_handle_index
    self.num_all_gather_handles = len(self.all_gather_handle_index_to_bucket_index_map)

    self.overlap_param_gather = self.config.overlap_param_gather
    self.remove_pre_hook_handle = None
    if self.overlap_param_gather:
        self.enable_pre_hook()

    self.update_successful = False

    # Update optimizer groups.
    # - Also, leverage state_dict() and load_state_dict() to
    #   recast preexisting per-param state tensors.
    self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
    self.optimizer.load_state_dict(self.optimizer.state_dict())


def _copy_model_grads_to_main_grads_zero3(self):
    """
    Copy model grads to main grads.

    Since this step follows a reduce-scatter through the DDP's grad
    buffer, this method is responsible for copying the updated grads
    from the grad buffer to the main shard's grad field.
    """

    # Utility method for copying group grads.
    def copy_group_grads(model_groups, shard_main_groups):
        for model_group, shard_main_group in zip(model_groups, shard_main_groups):
            for model_param, shard_main_param in zip(model_group, shard_main_group):

                if hasattr(model_param, 'enable_zero3') and model_param.enable_zero3:
                    model_grad = model_param.main_grad
                    shard_model_grad = model_grad.view(-1)
                    shard_main_param.grad = shard_model_grad.float()
                    continue
                param_range_map = self._get_model_param_range_map(model_param)
                param_range = param_range_map["param"]
                assert param_range.size == shard_main_param.nelement()   
                model_grad = model_param.main_grad
                shard_model_grad = model_grad.view(-1)[param_range.start : param_range.end]
                shard_main_param.grad = shard_model_grad.float()

    # Copy model groups to shard groups.
    copy_group_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups)
    copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)


def reuse_fp32_param_distrib_optimizer_init_wrapper(init_func):
    @wraps(init_func)
    def reuse_fp32_param_distrib_optimizer_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        global_args = get_args()
        self.reuse_fp32_param = global_args.reuse_fp32_param if hasattr(global_args, "reuse_fp32_param") else False
        # A flag that disables the value subtraction when the `fp16_tensor_convert_to_fp32_tensor` function is invoked for the first time.
        self.first_sub_flag = True
        if self.reuse_fp32_param:
            from mindspeed.op_builder import AlgorithmOpBuilder
            reuse_data_ptr = AlgorithmOpBuilder().load().reuse_data_ptr
            data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
            data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
            self.model_param_bucket_and_res_map = {}
            self.model_param_bucket_and_shard_main_param_int32_view_map = {}
            self.shard_main_param_res_buffers = []
            self.bucket_num_groups = []
            if data_parallel_world_size == 1:
                self.shard_fp32_param_fp16_view_group = []
                for buffer in self.buffers:
                    buffer_numel = buffer.param_data.numel()
                    shard_res_and_buffer_model_param = torch.zeros(buffer_numel * 2, dtype=torch.bfloat16, device=buffer.param_data.device)
                    shard_main_param_int32_view_buffer = torch.empty(buffer_numel, dtype=torch.int32, device=buffer.param_data.device)
                    reuse_data_ptr(shard_main_param_int32_view_buffer, shard_res_and_buffer_model_param, 0)
                    self.shard_main_param_res_buffers.append(shard_res_and_buffer_model_param)
                    self.model_param_bucket_and_shard_main_param_int32_view_map[shard_res_and_buffer_model_param] = shard_main_param_int32_view_buffer
                for model_fp16_params_this_group, shard_fp32_from_float16_group in zip(
                    self.model_float16_groups, self.shard_fp32_from_float16_groups):
                    for i, (model_param, shard_fp32_main_param) in enumerate(
                        zip(model_fp16_params_this_group, shard_fp32_from_float16_group)):
                        gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                        data_start_index, data_end_index, bucket_id = self.buffers[gbuf_index].param_index_map[model_param]
                        reuse_data_ptr(shard_fp32_from_float16_group[i], self.shard_main_param_res_buffers[gbuf_index], data_start_index)
                        old_param_data = model_param.data
                        model_param.data = self.shard_main_param_res_buffers[gbuf_index][data_start_index + data_end_index: 2 * data_end_index].view(old_param_data.shape)
                        model_param.data.detach().copy_(old_param_data)
                        self.shard_fp32_param_fp16_view_group.append(self.shard_main_param_res_buffers[gbuf_index][2 * data_start_index: 2 * data_end_index])
                for i, buffer in enumerate(self.buffers):
                    buffer_numel = buffer.param_data.numel()
                    reuse_data_ptr(buffer.param_data, self.shard_main_param_res_buffers[i], buffer_numel)
            else:
                for buffer in self.buffers:
                    self.bucket_num_group = []
                    bucket_res_numel = 0
                    res_numel = buffer.numel // data_parallel_world_size
                    shard_main_param_res_buffer = torch.zeros(res_numel, dtype=torch.bfloat16, device=buffer.param_data.device)
                    self.shard_main_param_res_buffers.append(shard_main_param_res_buffer)
                    for bucket in buffer.buckets:
                        self.bucket_num_group.append(bucket.param_data.numel())
                        param_data_dp_numel = bucket.param_data.numel() // data_parallel_world_size
                        shard_main_param_int32_view_bucket = torch.empty(param_data_dp_numel, dtype=torch.int32, device=bucket.param_data.device)
                        reuse_data_ptr(
                            shard_main_param_int32_view_bucket,
                            buffer.param_data,
                            (bucket_res_numel * data_parallel_world_size) // 2 + max(0, data_parallel_rank - 1) * param_data_dp_numel // 2)
                        self.model_param_bucket_and_res_map[bucket.param_data] = self.shard_main_param_res_buffers[-1][bucket_res_numel: bucket_res_numel + param_data_dp_numel]
                        self.model_param_bucket_and_shard_main_param_int32_view_map[bucket.param_data] = shard_main_param_int32_view_bucket
                        bucket_res_numel += param_data_dp_numel
                    self.bucket_num_groups.append(self.bucket_num_group)
                for model_fp16_params_this_group, shard_fp32_from_float16_group in zip(
                    self.model_float16_groups, self.shard_fp32_from_float16_groups):
                    for i, (model_param, shard_fp32_main_param) in enumerate(
                        zip(model_fp16_params_this_group, shard_fp32_from_float16_group)):
                        world_range = self._get_model_param_range_map(model_param)["gbuf_world_in_bucket"]
                        gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                        model_param_buffer = self.buffers[gbuf_index].param_data
                        bucket_offset_in_buffer = sum(self.bucket_num_groups[gbuf_index][:bucket_id]) // 2
                        model_param_bucket = self.buffers[gbuf_index].buckets[bucket_id].param_data
                        model_param_bucket_numel_per_dp = model_param_bucket.numel() // data_parallel_world_size
                        shard_fp32_param_bucket_offset = world_range.start if data_parallel_rank == 0 else \
                            world_range.start - model_param_bucket_numel_per_dp * (1 + data_parallel_rank) // 2
                        shard_main_param_buffer_start = bucket_offset_in_buffer + shard_fp32_param_bucket_offset
                        reuse_data_ptr(shard_fp32_from_float16_group[i], model_param_buffer, shard_main_param_buffer_start)
            torch_npu.npu.empty_cache()
            self._copy_model_params_to_main_params = _copy_model_params_to_main_params
            self.load_parameter_state_from_dp_zero_func = self.load_parameter_state_from_dp_zero
            self.load_parameter_state_from_dp_zero = types.MethodType(load_parameter_state_from_dp_zero, self)
            self.get_parameter_state_dp_zero_func = self.get_parameter_state_dp_zero
            self.get_parameter_state_dp_zero = types.MethodType(get_parameter_state_dp_zero, self)
            self.fp16_tensor_convert_to_fp32_tensor = types.MethodType(fp16_tensor_convert_to_fp32_tensor, self)
            self.fp32_tensor_convert_to_fp16_tensor = types.MethodType(fp32_tensor_convert_to_fp16_tensor, self)    
    return reuse_fp32_param_distrib_optimizer_init


def _copy_model_params_to_main_params():
    pass


def ema_distrib_optimizer_init_wrapper(init_func):
    @wraps(init_func)
    def ema_distrib_optimizer_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        self.load_parameter_state_from_dp_zero_func_temp = self.load_parameter_state_from_dp_zero
        self.load_parameter_state_from_dp_zero = types.MethodType(load_ema_state_from_dp_zero, self)
        self.get_parameter_state_dp_zero_func_temp = self.get_parameter_state_dp_zero
        self.get_parameter_state_dp_zero = types.MethodType(get_ema_state_dp_zero, self)
    return ema_distrib_optimizer_init


def load_ema_state_from_dp_zero(self, state_dict):
    """Load parameter state (i.e., parameter & optimizer tensors) from DP 0 rank,
    using the new checkpoint format with coalesced state across buckets.

    This method performs the reverse of get_parameter_state_dp_zero():
    - Scatter contiguous buffers from DP rank 0 to each DP rank (each DP
    rank receives its relevant subset of the world buffers).
    - For each DP rank, copy param & optimizer shards from contiguous CPU
    buffers. (e.g., one buffer each for main_param, exp_avg, and
    exp_avg_sq).
    """
    self.load_parameter_state_from_dp_zero_func_temp(state_dict)
    # Data parallelism variables.
    data_parallel_world_size = self.data_parallel_group_gloo.size()
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    data_parallel_group_gloo = self.data_parallel_group_gloo
    data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
        self.data_parallel_group_gloo
    )

    # Scatter tensors to all DP ranks.
    for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
        for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
            if data_parallel_rank == 0:
                buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
                checkpoint_numel_unpadded = state_dict[gbuf_idx][dtype]["numel_unpadded"]
                assert buffer_numel_unpadded == checkpoint_numel_unpadded, (
                    f"Number of unpadded elements must be same in current run "
                    f"({buffer_numel_unpadded}) and checkpoint ({checkpoint_numel_unpadded})"
                )
            for key in ("ema_params",):
                offset_in_world_tensors = 0
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    # Compute local DP contiguous shard's size.
                    gbuf_world_numel = (
                        self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    )
                    assert gbuf_world_numel % data_parallel_world_size == 0
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size
                    gbuf_world_numel_unpadded = (
                        self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                    )
                    assert gbuf_world_numel_unpadded <= gbuf_world_numel

                    # Contiguous local shards (received from DP rank 0).
                    recv_tensor = torch.empty(
                        (gbuf_local_numel,), dtype=torch.float32, device="cpu"
                    )

                    # Scatter tensor list.
                    if data_parallel_rank == 0:
                        world_tensors = state_dict[gbuf_idx][dtype][key]

                        start = offset_in_world_tensors
                        end = offset_in_world_tensors + gbuf_world_numel_unpadded
                        assert 0 <= start < end <= world_tensors.numel()
                        world_tensor = world_tensors[start:end]
                        offset_in_world_tensors += gbuf_world_numel_unpadded

                        # Pad world_tensor to gbuf_world_numel. Don't pad at the front, pad at the back.
                        world_tensor = torch.nn.functional.pad(
                            world_tensor, (0, gbuf_world_numel - gbuf_world_numel_unpadded)
                        )
                        assert world_tensor.numel() == gbuf_world_numel
                        gbuf_start_idxs = list(range(0, gbuf_world_numel, gbuf_local_numel))
                        send_tensors = [
                            world_tensor[i: (i + gbuf_local_numel)] for i in gbuf_start_idxs
                        ]
                    else:
                        send_tensors = None

                    # Scatter.
                    if get_args().disable_gloo_group:
                        from mindspeed.utils import _scatter_hccl
                        _scatter_hccl(
                            recv_tensor,
                            send_tensors,
                            data_parallel_global_ranks[0],
                            self.data_parallel_group)
                    else:
                        torch.distributed.scatter(
                            recv_tensor,
                            send_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )

                    # Copy local contiguous shards to param/optim shards.
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][
                            group_order
                        ]

                        optim_state = self.optimizer.state[main_param]
                        if key not in self.optimizer.state[main_param].keys():
                            optim_state[key] = main_param.clone()
                        tensor_to_copy_into = optim_state[key]

                        # Copy states into contiguous shard.
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end
                        tensor_to_copy_into.data.copy_(
                            recv_tensor[gbuf_local_start:gbuf_local_end]
                        )


def get_ema_state_dp_zero(self):
    """Get parameter state (i.e., parameter & optimizer tensors).

    This method performs two steps:
    - For each DP rank, copy param & optimizer shards to contiguous CPU
    buffers (e.g., one buffer each for main_param, exp_avg, and
    exp_avg_sq).
    - Gather contiguous buffers on DP rank 0 and concatenate to world
    buffers.
    """
    state = self.get_parameter_state_dp_zero_func_temp()
    # Data parallelism variables.
    data_parallel_world_size = self.data_parallel_group_gloo.size()
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    data_parallel_group_gloo = self.data_parallel_group_gloo
    data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
        self.data_parallel_group_gloo
    )
    for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):

        # Iterate grad buffers (by data type).
        dtype_state = state[gbuf_idx]
        assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
        for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
            buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
            # Create coalesced tensors for all state related to parameters in this buffer.
            world_tensors = {}
            if data_parallel_rank == 0:
                world_tensors = {
                    key: torch.empty(
                        (buffer_numel_unpadded,), dtype=torch.float32, device="cpu"
                    )
                    for key in ("ema_params",)
                }
                world_tensors["numel_unpadded"] = buffer_numel_unpadded
            offset_in_world_tensors = 0
            for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):

                # Compute local DP contiguous shard's size.
                gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                assert gbuf_world_numel % data_parallel_world_size == 0
                gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                gbuf_world_numel_unpadded = (
                    self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                )
                assert gbuf_world_numel_unpadded <= gbuf_world_numel

                local_shards = {
                    key: torch.empty((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                    for key in ("ema_params", )
                }

                # Build contiguous DP rank shards (for param + optim states).
                for model_param, param_range_map in gbuf_range_map["param_map"].items():

                    # Main param & optimizer states.
                    group_index, group_order = self.model_param_group_index_map[model_param]
                    main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                    optim_state = self.optimizer.state[main_param]

                    tensors = {
                        "param": main_param,
                        **optim_state,
                    }

                    # Copy states into contiguous shard.
                    gbuf_local_start = param_range_map["gbuf_local"].start
                    gbuf_local_end = param_range_map["gbuf_local"].end
                    for key in local_shards:
                        local_shards[key][gbuf_local_start:gbuf_local_end].data.copy_(
                            tensors[key].detach().cpu()
                        )

                # Gather contiguous shards on DP rank 0.
                for key, send_tensor in local_shards.items():

                    # Gather tensor list.
                    if data_parallel_rank == 0:
                        recv_tensors = [
                            torch.empty((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                            for _ in range(data_parallel_world_size)
                        ]
                    else:
                        recv_tensors = None

                    # Gather.
                    if get_args().disable_gloo_group:
                        from mindspeed.utils import _gather_hccl
                        _gather_hccl(
                            send_tensor,
                            recv_tensors,
                            self.data_parallel_group,
                        )
                    else:
                        torch.distributed.gather(
                            send_tensor,
                            recv_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )

                    # Concatenate.
                    if data_parallel_rank == 0:
                        recv_tensors_concatenated = torch.cat(recv_tensors)
                        # Copy this bucket's collected all-gather tensors into the right place in the
                        # tensor for the buffer. The tensor for the buffer gets rid of the padding
                        # between buckets.
                        start = offset_in_world_tensors
                        end = offset_in_world_tensors + gbuf_world_numel_unpadded
                        world_tensors[key][start:end].copy_(
                            recv_tensors_concatenated[:gbuf_world_numel_unpadded]
                        )

                offset_in_world_tensors += gbuf_world_numel_unpadded

            # Collect world state.
            dtype_state[dtype].update(world_tensors)
        state[gbuf_idx] = dtype_state

    return state



def load_parameter_state_from_dp_zero(self, state_dict):
    self.load_parameter_state_from_dp_zero_func(state_dict)
    self.first_sub_flag = False
    data_parallel_world_size = self.data_parallel_group_gloo.size()
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    data_parallel_group_gloo = self.data_parallel_group_gloo
    data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
        self.data_parallel_group_gloo
    )
    if data_parallel_world_size == 1 or \
        not hasattr(self, "shard_main_param_res_buffers"):
        return
    for i, shard_main_param_res_buffer in enumerate(self.shard_main_param_res_buffers):
        shard_res_numel = shard_main_param_res_buffer.numel()
        recv_tensor = torch.empty((shard_res_numel,), dtype=torch.float16, device="cpu")
        if data_parallel_rank == 0:
            send_tensors = [
                state_dict["shard_main_param_res"][i][
                    dpr * shard_res_numel: (dpr + 1) * shard_res_numel] for dpr in range(data_parallel_world_size)
            ]
        else:
            send_tensors = None

        if get_args().disable_gloo_group:
            from mindspeed.utils import _scatter_hccl
            _scatter_hccl(
                recv_tensor,
                send_tensors,
                data_parallel_global_ranks[0],
                self.data_parallel_group)
        else:
            torch.distributed.scatter(
                recv_tensor,
                send_tensors,
                data_parallel_global_ranks[0],
                data_parallel_group_gloo,
            )
        recv_tensor_bf16_view = torch.tensor(recv_tensor.data.untyped_storage(), dtype=torch.bfloat16, device=recv_tensor.device)
        shard_main_param_res_buffer.copy_(recv_tensor_bf16_view)


def get_parameter_state_dp_zero(self):
    state = self.get_parameter_state_dp_zero_func()
    data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    data_parallel_group_gloo = self.data_parallel_group_gloo
    data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
        self.data_parallel_group_gloo
    )
    if data_parallel_world_size == 1 or not hasattr(self, "shard_main_param_res_buffers"):
        return state
   
    # gather buffer res
    buffer_res_full_shard = []
    for shard_main_param_res_buffer in self.shard_main_param_res_buffers:
        if get_args().disable_gloo_group:
            recv_tensors = [torch.empty(shard_main_param_res_buffer.numel(), dtype=torch.float16, device="cpu") for _
                            in range(data_parallel_world_size)]
        else:
            if data_parallel_rank == 0:
                recv_tensors = [torch.empty((shard_main_param_res_buffer.numel(),), dtype=torch.float16, device="cpu") for _ in range(data_parallel_world_size)]
            else:
                recv_tensors = None
       
        send_tensor = torch.empty((shard_main_param_res_buffer.numel(),), dtype=torch.float16, device="cpu")
        send_tensor_bf16_view = torch.tensor(send_tensor.data.untyped_storage(), dtype=torch.bfloat16, device=send_tensor.device)
        send_tensor_bf16_view.copy_(shard_main_param_res_buffer.detach().cpu())
        if get_args().disable_gloo_group:
            from mindspeed.utils import _gather_hccl
            _gather_hccl(
                send_tensor,
                recv_tensors,
                self.data_parallel_group,
            )
        else:
            torch.distributed.gather(
                send_tensor,
                recv_tensors,
                data_parallel_global_ranks[0],
                data_parallel_group_gloo,
            )
        if data_parallel_rank == 0:
            buffer_res_full_shard.append(torch.cat(recv_tensors))
   
    state['shard_main_param_res'] = buffer_res_full_shard
    return state


def fp16_tensor_convert_to_fp32_tensor(self):
    """
    res(0000) + bf16(pppp) -> fp32(0p0p0p0p)

    Transform the bf16 data and residuals data in the continuous memory block
    into the fp32 tensor through view transposition.
    """
    data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    iteration = getattr(get_args(), "iteration", 0)
    npu_deterministic = getattr(get_args(), "npu_deterministic", False)
    if data_parallel_world_size == 1:
        for shard_fp32_param_fp16_view in self.shard_fp32_param_fp16_view_group:
            shard_fp32_param_fp16_view.copy_(
                    shard_fp32_param_fp16_view.view(2, -1).transpose(1, 0).reshape(-1).contiguous())
       
        if npu_deterministic:
            if not self.first_sub_flag:
                fp16_tensor_convert_to_fp32_tensor_deterministic(self.shard_fp32_from_float16_groups, self.optimizer)
        else:
            for shard_res_and_buffer_model_param in self.shard_main_param_res_buffers:
                shard_main_param_int32_view_buffer = self.model_param_bucket_and_shard_main_param_int32_view_map[shard_res_and_buffer_model_param]
                if not self.first_sub_flag:
                    shard_main_param_int32_view_buffer.sub_(32768)
    else:
        for buffer in self.buffers:
            for bucket in buffer.buckets:
                bucket_param_data = bucket.param_data
                param_data_dp_numel = bucket_param_data.numel() // data_parallel_world_size
                bucket_res = self.model_param_bucket_and_res_map[bucket_param_data]
                if data_parallel_rank == 0:
                    bucket_param_data[param_data_dp_numel:param_data_dp_numel * 2].copy_(bucket_param_data[:param_data_dp_numel])
                bucket_res_position = max(0, data_parallel_rank - 1) * param_data_dp_numel
                shard_fp32_main_param_view = bucket_param_data[bucket_res_position: bucket_res_position + param_data_dp_numel * 2]
                shard_main_param_int32_view_bucket = self.model_param_bucket_and_shard_main_param_int32_view_map[bucket_param_data]

                loops = param_data_dp_numel // TRANSPOSE_BF16_BLOCK_SIZE
                remain = param_data_dp_numel % TRANSPOSE_BF16_BLOCK_SIZE
                workspace = torch.zeros(
                    TRANSPOSE_BF16_BLOCK_SIZE * 2, dtype=torch.bfloat16, device=bucket_res.device)
                residual_space = bucket_res
                bf16_space_dp_rank = max(1, data_parallel_rank)
                bf16_space = bucket_param_data[param_data_dp_numel * bf16_space_dp_rank :param_data_dp_numel * (bf16_space_dp_rank + 1)]
           
                for loop in range(loops):
                    copy_start = loop * TRANSPOSE_BF16_BLOCK_SIZE
                    copy_end = (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE
                    workspace_convert_view = workspace[:TRANSPOSE_BF16_BLOCK_SIZE * 2]
                    workspace[:TRANSPOSE_BF16_BLOCK_SIZE].copy_(residual_space[copy_start: copy_end])
                    workspace[TRANSPOSE_BF16_BLOCK_SIZE:TRANSPOSE_BF16_BLOCK_SIZE * 2].copy_(bf16_space[copy_start: copy_end])
                    shard_fp32_main_param_view[copy_start * 2: copy_end * 2].copy_(
                        workspace_convert_view.view(2, -1).transpose(1, 0).reshape(-1).contiguous())

                if remain > 0:
                    workspace_convert_view = workspace[:remain * 2]
                    workspace[:remain].copy_(residual_space[-remain:])
                    workspace[remain:remain * 2].copy_(bf16_space[-remain:])
                    shard_fp32_main_param_view[-remain * 2:].copy_(
                        workspace_convert_view.view(2, -1).transpose(1, 0).reshape(-1).contiguous())
           
                if not self.first_sub_flag and not npu_deterministic:
                    shard_main_param_int32_view_bucket[:param_data_dp_numel].sub_(32768)
       
        if not self.first_sub_flag and npu_deterministic:
            fp16_tensor_convert_to_fp32_tensor_deterministic(self.shard_fp32_from_float16_groups, self.optimizer)


def fp32_tensor_convert_to_fp16_tensor(self):
    """
    fp32(0p0p0p0p) -> fp32(0'p0'p0'p0'p) -> res(0000) + bf16(pppp)

    Transform the fp32 tensor in the continuous memory block
    into the bf16 data and residual through view transposition.
    """
    data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    npu_deterministic = getattr(get_args(), "npu_deterministic", False)
    if data_parallel_world_size == 1:
        if npu_deterministic:
            fp32_tensor_convert_to_fp16_tensor_deterministic(self.shard_fp32_from_float16_groups, self.optimizer)
        else:
            for shard_res_and_buffer_model_param in self.shard_main_param_res_buffers:
                shard_main_param_int32_view_buffer = self.model_param_bucket_and_shard_main_param_int32_view_map[shard_res_and_buffer_model_param]
                shard_main_param_int32_view_buffer.add_(32768)
        self.first_sub_flag = False

        for shard_fp32_param_fp16_view in self.shard_fp32_param_fp16_view_group:
            shard_fp32_param_fp16_view.copy_(
                    shard_fp32_param_fp16_view.view(-1, 2).transpose(1, 0).reshape(-1).contiguous())
    else:
        if npu_deterministic:
            fp32_tensor_convert_to_fp16_tensor_deterministic(self.shard_fp32_from_float16_groups, self.optimizer)
        else:
            for buffer in self.buffers:
                for bucket in buffer.buckets:
                    bucket_param_data = bucket.param_data
                    param_data_dp_numel = bucket_param_data.numel() // data_parallel_world_size
                    shard_main_param_int32_view_bucket = self.model_param_bucket_and_shard_main_param_int32_view_map[bucket_param_data]
                    shard_main_param_int32_view_bucket[:param_data_dp_numel].add_(32768)

        for buffer in self.buffers:
            for bucket in buffer.buckets:
                self.first_sub_flag = False
                bucket_param_data = bucket.param_data
                param_data_dp_numel = bucket_param_data.numel() // data_parallel_world_size
                bucket_res = self.model_param_bucket_and_res_map[bucket_param_data]
                bucket_res_position = max(0, data_parallel_rank - 1) * param_data_dp_numel
                shard_fp32_main_param_view = bucket_param_data[bucket_res_position: bucket_res_position + param_data_dp_numel * 2]

                loops = param_data_dp_numel // TRANSPOSE_BF16_BLOCK_SIZE
                remain = param_data_dp_numel % TRANSPOSE_BF16_BLOCK_SIZE
                workspace = torch.zeros(
                    TRANSPOSE_BF16_BLOCK_SIZE * 2, dtype=torch.bfloat16, device=bucket_res.device)
                bf16_space_dp_rank = max(0, data_parallel_rank - 1)
                residual_space = bucket_res
                bf16_space = bucket_param_data[
                    param_data_dp_numel * bf16_space_dp_rank :param_data_dp_numel * (bf16_space_dp_rank + 1)]
           
                for loop in range(loops):
                    workspace_convert_view = workspace[:TRANSPOSE_BF16_BLOCK_SIZE * 2]
                    workspace_convert_view.copy_(
                        shard_fp32_main_param_view[loop * TRANSPOSE_BF16_BLOCK_SIZE * 2: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE * 2])
                    temp = workspace_convert_view.view(-1, 2).transpose(1, 0).reshape(-1).contiguous()
                    residual_space[loop * TRANSPOSE_BF16_BLOCK_SIZE: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE].copy_(
                        temp[:TRANSPOSE_BF16_BLOCK_SIZE])
                    bf16_space[loop * TRANSPOSE_BF16_BLOCK_SIZE: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE].copy_(
                        temp[TRANSPOSE_BF16_BLOCK_SIZE: TRANSPOSE_BF16_BLOCK_SIZE * 2])
           
                if remain > 0:
                    workspace_convert_view = workspace[:remain * 2]
                    workspace_convert_view.copy_(shard_fp32_main_param_view[-remain * 2:])
                    temp = workspace_convert_view.view(-1, 2).transpose(1, 0).reshape(-1).contiguous()
                    residual_space[-remain:].copy_(temp[:remain])
                    bf16_space[-remain:].copy_(temp[remain: remain * 2])

                if data_parallel_rank != 0:
                    shard_fp32_main_param_view[param_data_dp_numel:param_data_dp_numel * 2].copy_(shard_fp32_main_param_view[:param_data_dp_numel])


def fp16_tensor_convert_to_fp32_tensor_deterministic(shard_fp32_from_float16_groups, optimizer):
    assert hasattr(optimizer, "state")
    for shard_fp32_from_float16_group in shard_fp32_from_float16_groups:
        for shard_fp32_param in shard_fp32_from_float16_group:
            if "exp_avg_sq" not in optimizer.state[shard_fp32_param]:
                continue
            shard_int32_tensor = shard_fp32_param.view(torch.int32)
            assert shard_int32_tensor.numel() == shard_fp32_param.numel()
            loops = shard_int32_tensor.numel() // TRANSPOSE_BF16_BLOCK_SIZE
            remain = shard_int32_tensor.numel() % TRANSPOSE_BF16_BLOCK_SIZE
            exp_avg_sq_flatten = optimizer.state[shard_fp32_param]["exp_avg_sq"].reshape(-1)
            for loop in range(loops):
                odd_even_tensor = torch.sign(exp_avg_sq_flatten[loop * TRANSPOSE_BF16_BLOCK_SIZE: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE] > 0)
                shard_int32_tensor[loop * TRANSPOSE_BF16_BLOCK_SIZE: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE].add_(odd_even_tensor)
            if remain > 0:
                odd_even_tensor = torch.sign(exp_avg_sq_flatten[-remain:] > 0)
                shard_int32_tensor[-remain:].add_(odd_even_tensor)
            shard_int32_tensor.sub_(32768)
            optimizer.state[shard_fp32_param]["exp_avg_sq"].abs_()


def fp32_tensor_convert_to_fp16_tensor_deterministic(shard_fp32_from_float16_groups, optimizer):
    assert hasattr(optimizer, "state")
    for shard_fp32_from_float16_group in shard_fp32_from_float16_groups:
        for shard_fp32_param in shard_fp32_from_float16_group:
            if "exp_avg_sq" not in optimizer.state[shard_fp32_param]:
                continue
            shard_int32_tensor = shard_fp32_param.view(torch.int32)
            assert shard_int32_tensor.numel() == shard_fp32_param.numel()
            loops = shard_int32_tensor.numel() // TRANSPOSE_BF16_BLOCK_SIZE
            remain = shard_int32_tensor.numel() % TRANSPOSE_BF16_BLOCK_SIZE
            exp_avg_sq_flatten = optimizer.state[shard_fp32_param]["exp_avg_sq"].reshape(-1)
            shard_int32_tensor.add_(32768)
            for loop in range(loops):
                odd_even_tensor = ((shard_int32_tensor[loop * TRANSPOSE_BF16_BLOCK_SIZE: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE] & 131071) == 65536).int()
                shard_int32_tensor[loop * TRANSPOSE_BF16_BLOCK_SIZE: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE].sub_(odd_even_tensor)
                sign_tensor = torch.sign(odd_even_tensor - 0.5)
                exp_avg_sq_flatten[loop * TRANSPOSE_BF16_BLOCK_SIZE: (loop + 1) * TRANSPOSE_BF16_BLOCK_SIZE].mul_(sign_tensor)
            if remain > 0:
                odd_even_tensor = ((shard_int32_tensor[-remain:] & 131071) == 65536).int()
                shard_int32_tensor[-remain:].sub_(odd_even_tensor)
                sign_tensor = torch.sign(odd_even_tensor - 0.5)
                exp_avg_sq_flatten[-remain:].mul_(sign_tensor)


def get_parameter_state_dp_zero_hccl(self):
    """
        Replace the communication method of gather from gloo to hccl.
    """

    # Data parallelism variables.
    data_parallel_world_size = self.data_parallel_group.size()
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group)
    data_parallel_group = self.data_parallel_group

    # Collect param states.
    state = {
        "buckets_coalesced": True,
    }
    for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):

        # Iterate grad buffers (by data type).
        dtype_state = {}
        assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
        for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
            buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
            # Create coalesced tensors for all state related to parameters in this buffer.
            world_tensors = {}
            if data_parallel_rank == 0:
                world_tensors = {
                    key: torch.zeros(
                        (buffer_numel_unpadded,), dtype=torch.float32, device="cpu"
                    )
                    for key in ("param", "exp_avg", "exp_avg_sq")
                }
                world_tensors["numel_unpadded"] = buffer_numel_unpadded
            offset_in_world_tensors = 0
            for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):

                # Compute local DP contiguous shard's size.
                gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                assert gbuf_world_numel % data_parallel_world_size == 0
                gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                gbuf_world_numel_unpadded = (
                    self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                )
                assert gbuf_world_numel_unpadded <= gbuf_world_numel

                local_shards = {
                    key: torch.zeros((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                    for key in ("param", "exp_avg", "exp_avg_sq")
                }

                # Build contiguous DP rank shards (for param + optim states).
                for model_param, param_range_map in gbuf_range_map["param_map"].items():

                    # Main param & optimizer states.
                    group_index, group_order = self.model_param_group_index_map[model_param]
                    main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                    optim_state = self.optimizer.state[main_param]

                    tensors = {
                        "param": main_param,
                        **optim_state,
                    }

                    # Copy states into contiguous shard.
                    gbuf_local_start = param_range_map["gbuf_local"].start
                    gbuf_local_end = param_range_map["gbuf_local"].end
                    for key in local_shards:
                        local_shards[key][gbuf_local_start:gbuf_local_end].data.copy_(
                            tensors[key].detach().cpu()
                        )

                # Gather contiguous shards on DP rank 0.
                for key, send_tensor in local_shards.items():

                    # Gather tensor list.
                    recv_tensors = [
                        torch.zeros((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                        for _ in range(data_parallel_world_size)
                    ]

                    # Gather.
                    from mindspeed.utils import _gather_hccl
                    _gather_hccl(
                        send_tensor,
                        recv_tensors,
                        data_parallel_group,
                    )

                    # Concatenate.
                    if data_parallel_rank == 0:
                        recv_tensors_concatenated = torch.cat(recv_tensors)
                        # Copy this bucket's collected all-gather tensors into the right place in the
                        # tensor for the buffer. The tensor for the buffer gets rid of the padding
                        # between buckets.
                        start = offset_in_world_tensors
                        end = offset_in_world_tensors + gbuf_world_numel_unpadded
                        world_tensors[key][start:end].copy_(
                            recv_tensors_concatenated[:gbuf_world_numel_unpadded]
                        )

                offset_in_world_tensors += gbuf_world_numel_unpadded

            # Collect world state.
            dtype_state[dtype] = world_tensors
        state[gbuf_idx] = dtype_state

    return state


def load_parameter_state_from_dp_zero_hccl(self, state_dict):
    """Load parameter state (i.e., parameter & optimizer tensors) from DP 0 rank,
    using the new checkpoint format with coalesced state across buckets.

    This method performs the reverse of get_parameter_state_dp_zero():
    - Scatter contiguous buffers from DP rank 0 to each DP rank (each DP
      rank receives its relevant subset of the world buffers).
    - For each DP rank, copy param & optimizer shards from contiguous CPU
      buffers. (e.g., one buffer each for main_param, exp_avg, and
      exp_avg_sq).
    """

    # Data parallelism variables.
    data_parallel_world_size = self.data_parallel_group.size()
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group)
    data_parallel_group = self.data_parallel_group
    data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
        self.data_parallel_group
    )

    # Scatter tensors to all DP ranks.
    for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
        for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
            if data_parallel_rank == 0:
                buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
                checkpoint_numel_unpadded = state_dict[gbuf_idx][dtype]["numel_unpadded"]
                assert buffer_numel_unpadded == checkpoint_numel_unpadded, (
                    f"Number of unpadded elements must be same in current run "
                    f"({buffer_numel_unpadded}) and checkpoint ({checkpoint_numel_unpadded})"
                )
            for key in ("param", "exp_avg", "exp_avg_sq"):
                offset_in_world_tensors = 0
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    # Compute local DP contiguous shard's size.
                    gbuf_world_numel = (
                        self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    )
                    assert gbuf_world_numel % data_parallel_world_size == 0
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size
                    gbuf_world_numel_unpadded = (
                        self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                    )
                    assert gbuf_world_numel_unpadded <= gbuf_world_numel

                    # Contiguous local shards (received from DP rank 0).
                    recv_tensor = torch.zeros(
                        (gbuf_local_numel,), dtype=torch.float32, device="cpu"
                    )

                    # Scatter tensor list.
                    if data_parallel_rank == 0:
                        world_tensors = state_dict[gbuf_idx][dtype][key]

                        start = offset_in_world_tensors
                        end = offset_in_world_tensors + gbuf_world_numel_unpadded
                        assert 0 <= start < end <= world_tensors.numel()
                        world_tensor = world_tensors[start:end]
                        offset_in_world_tensors += gbuf_world_numel_unpadded

                        # Pad world_tensor to gbuf_world_numel. Don't pad at the front, pad at the back.
                        world_tensor = torch.nn.functional.pad(
                            world_tensor, (0, gbuf_world_numel - gbuf_world_numel_unpadded)
                        )
                        assert world_tensor.numel() == gbuf_world_numel
                        gbuf_start_idxs = list(range(0, gbuf_world_numel, gbuf_local_numel))
                        send_tensors = [
                            world_tensor[i: (i + gbuf_local_numel)] for i in gbuf_start_idxs
                        ]
                    else:
                        send_tensors = None

                    # Scatter.
                    from mindspeed.utils import _scatter_hccl
                    _scatter_hccl(
                        recv_tensor,
                        send_tensors,
                        data_parallel_global_ranks[0],
                        data_parallel_group)

                    # Copy local contiguous shards to param/optim shards.
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][
                            group_order
                        ]
                        if key == "param":
                            tensor_to_copy_into = main_param
                        else:
                            optim_state = self.optimizer.state[main_param]
                            tensor_to_copy_into = optim_state[key]

                        # Copy states into contiguous shard.
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end
                        tensor_to_copy_into.data.copy_(
                            recv_tensor[gbuf_local_start:gbuf_local_end]
                        )
