# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import math
import types
from functools import wraps
import torch
import torch_npu

from megatron.training import get_args

from mindspeed.optimizer.distrib_optimizer import (
    get_parameter_state_dp_zero,
    load_parameter_state_from_dp_zero,
)
from .reuse_optimizer import _copy_model_params_to_main_params 
from .reuse_distrib_optimizer import (
    fp16_tensor_convert_to_fp32_tensor_dis,
    fp32_tensor_convert_to_fp16_tensor_dis,
)
from .reuse_optimizer import ConvertFp32BF16


@torch.no_grad()
def prepare_grads(self) -> bool:
    """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
    timers = self.config.timers

    # Copy gradients from model params to main params.
    if timers is not None:
        timers('optimizer-copy-to-main-grad', log_level=1).start(
            barrier=self.config.barrier_with_L1_time
        )
    self._copy_model_grads_to_main_grads()
    if timers is not None:
        timers('optimizer-copy-to-main-grad').stop()

    if self.config.reuse_fp32_param:
        # bf16 -> fp32
        self.fp16_tensor_convert_to_fp32_tensor()

    # Do unscale, check for inf, and update grad scaler only for
    # the case that grad scaler is provided.
    if self.grad_scaler:

        # Unscale and check for inf/nan.
        if timers is not None:
            timers('optimizer-unscale-and-check-inf', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        found_inf_flag = self._unscale_main_grads_and_check_for_nan()
        if timers is not None:
            timers('optimizer-unscale-and-check-inf').stop()

        # We are done with scaling gradients
        # so we can update the loss scale.
        self.grad_scaler.update(found_inf_flag)

        return found_inf_flag

    return False


@torch.no_grad()
def step_with_ready_grads(self) -> bool:
    """Step the optimizer with ready gradients, return successful."""
    timers = self.config.timers
    # Step the optimizer.
    if timers is not None:
        timers('optimizer-inner-step', log_level=1).start(
            barrier=self.config.barrier_with_L1_time
        )
    self.optimizer.step()
    if timers is not None:
        timers('optimizer-inner-step').stop()

    # Update params from main params.
    if timers is not None:
        timers('optimizer-copy-main-to-model-params', log_level=1).start(
            barrier=self.config.barrier_with_L1_time
        )
    if self.config.reuse_fp32_param:
        # fp32 -> bf16 + res
        self.fp32_tensor_convert_to_fp16_tensor()
    else:
        self._copy_main_params_to_model_params()
    if timers is not None:
        timers('optimizer-copy-main-to-model-params').stop()

    return True


# distrib
def reuse_fp32_param_distrib_optimizer_init_wrapper(init_func):
    @wraps(init_func)
    def reuse_fp32_param_distrib_optimizer_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        global_args = get_args()
        self.reuse_fp32_param = getattr(global_args, "reuse_fp32_param", False)
        self.disable_gloo_group = getattr(global_args, "disable_gloo_group", False)
        self.npu_deterministic = getattr(global_args, "npu_deterministic", False)
        # A flag that disables the value subtraction when the `fp16_tensor_convert_to_fp32_tensor` function is invoked for the first time.
        self.first_sub_flag = True

        if self.reuse_fp32_param:
            data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
            self.model_param_bucket_and_res_map = {}
            self.model_param_bucket_and_shard_main_param_int32_view_map = {}
            self.shard_main_param_res_buffers = []
            self.bucket_num_groups = []
            if data_parallel_world_size == 1:
                reuse_buffer_single(self)
            else:
                reuse_buffer_dis(self, data_parallel_world_size)
            torch_npu.npu.empty_cache()

            self._copy_model_params_to_main_params = _copy_model_params_to_main_params
            self.load_parameter_state_from_dp_zero_func = self.load_parameter_state_from_dp_zero
            self.load_parameter_state_from_dp_zero = types.MethodType(load_parameter_state_from_dp_zero, self)
            self.get_parameter_state_dp_zero_func = self.get_parameter_state_dp_zero
            self.get_parameter_state_dp_zero = types.MethodType(get_parameter_state_dp_zero, self)
            self.fp16_tensor_convert_to_fp32_tensor = types.MethodType(fp16_tensor_convert_to_fp32_tensor_dis, self)
            self.fp32_tensor_convert_to_fp16_tensor = types.MethodType(fp32_tensor_convert_to_fp16_tensor_dis, self)    
    return reuse_fp32_param_distrib_optimizer_init


def reuse_buffer_single(self):
    from mindspeed.op_builder import AlgorithmOpBuilder
    reuse_data_ptr = AlgorithmOpBuilder().load().reuse_data_ptr
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


def reuse_buffer_dis(self, data_parallel_world_size):
    from mindspeed.op_builder import AlgorithmOpBuilder
    reuse_data_ptr = AlgorithmOpBuilder().load().reuse_data_ptr
    if not self.disable_gloo_group:
        data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    else:
        data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group)
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


def optimizer_config_init_wrapper(init_func):
    @wraps(init_func)
    def optimizer_config_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        args = get_args()
        self.reuse_fp32_param = getattr(args, "reuse_fp32_param", False)

    return optimizer_config_init


def reuse_fp32_param_init_wrapper(init_func):
    @wraps(init_func)
    def reuse_fp32_param_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        args = get_args()
        self.reuse_fp32_param = getattr(args, "reuse_fp32_param", False)
        self.convert_func = ConvertFp32BF16()
        if self.reuse_fp32_param:
            self.res_float16_groups = []
            self.float16_float32_groups = []
            self.int32_float32_groups = []
            for float16_params_this_group, fp32_from_float16_group in zip(self.float16_groups, self.fp32_from_float16_groups):
                res_float16_params_this_group = []
                float16_float32_params_this_group = []
                int32_float32_params_this_group = []
                for i, (_, fp32_from_fp16_param) in enumerate(zip(float16_params_this_group, fp32_from_float16_group)):
                    res_float16_params_this_group.append(
                        torch.empty((fp32_from_fp16_param.numel() * 1), dtype=torch.bfloat16, device=fp32_from_fp16_param.device))
                    float16_float32_params_this_group.append(
                        torch.empty((fp32_from_fp16_param.numel() * 2), dtype=torch.bfloat16, device=fp32_from_fp16_param.device))
                    int32_float32_params_this_group.append(
                        torch.empty((fp32_from_fp16_param.numel() * 1), dtype=torch.int32, device=fp32_from_fp16_param.device))
                    self.convert_func.init_and_reuse_storage_of_tensors(fp32_from_float16_group[i],  
                                float16_float32_params_this_group[-1],
                                res_float16_params_this_group[-1],
                                float16_params_this_group[i],
                                int32_float32_params_this_group[-1]
                        )
                self.res_float16_groups.append(res_float16_params_this_group)
                self.float16_float32_groups.append(float16_float32_params_this_group)
                self.int32_float32_groups.append(int32_float32_params_this_group)
            self._copy_model_params_to_main_params = _copy_model_params_to_main_params
            if args.npu_deterministic:
                self.fp16_tensor_convert_to_fp32_tensor = types.MethodType(self.convert_func.fp16_tensor_convert_to_fp32_tensor_deterministic, self)
                self.fp32_tensor_convert_to_fp16_tensor = types.MethodType(self.convert_func.fp32_tensor_convert_to_fp16_tensor_deterministic, self)    
            else:
                self.fp16_tensor_convert_to_fp32_tensor = types.MethodType(self.convert_func.fp16_tensor_convert_to_fp32_tensor, self)
                self.fp32_tensor_convert_to_fp16_tensor = types.MethodType(self.convert_func.fp32_tensor_convert_to_fp16_tensor, self)    
    return reuse_fp32_param_init


def reuse_fp32_param_param_and_grad_buffer_init_wrapper(init_func):
    @wraps(init_func)
    def reuse_fp32_param_param_and_grad_buffer_init(*args, **kwargs):
        global_args = get_args()
        math_ceil = math.ceil
        if global_args.reuse_fp32_param and global_args.use_distributed_optimizer:
            def ceil_even(x):
                return math_ceil(math_ceil(x) / 2) * 2
            math.ceil = ceil_even
        init_func(*args, **kwargs)
        if global_args.reuse_fp32_param and global_args.use_distributed_optimizer:
            math.ceil = math_ceil
    return reuse_fp32_param_param_and_grad_buffer_init