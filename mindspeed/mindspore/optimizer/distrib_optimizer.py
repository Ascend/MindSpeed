# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Bytedance Inc. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import os
from textwrap import wrap
import types
from functools import wraps
import torch
import torch_npu
import torch.distributed

from megatron.training import get_args
from megatron.core.distributed.param_and_grad_buffer import BufferType

from mindspeed.optimizer.distrib_optimizer import _copy_model_params_to_main_params, load_parameter_state_from_dp_zero, \
    get_parameter_state_dp_zero, fp16_tensor_convert_to_fp32_tensor, fp32_tensor_convert_to_fp16_tensor

TRANSPOSE_BF16_BLOCK_SIZE = 4096 * 4096


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
            from mindspeed.mindspore.op_builder import AlgorithmOpBuilder
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
                        del old_param_data
                        self.shard_fp32_param_fp16_view_group.append(self.shard_main_param_res_buffers[gbuf_index][2 * data_start_index: 2 * data_end_index])
                for i, buffer in enumerate(self.buffers):
                    buffer_numel = buffer.param_data.numel()
                    reuse_data_ptr(buffer.param_data, self.shard_main_param_res_buffers[i], buffer_numel)
                    # fix mem by moving bucket.param_data manually
                    for bucket_id, bucket in enumerate(buffer.buckets):  # fix reuse
                        (start_index, end_index) = buffer.bucket_indices[bucket_id]
                        bucket.param_data.assign_value(buffer._get(
                            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
                        ))

                    # fix mem by moving shard_model_param manually
                    for model_float16_params_this_group, shard_float16_params_this_group in zip(
                            self.model_float16_groups, self.shard_float16_groups):
                        for model_param, shard_model_param in zip(model_float16_params_this_group,
                                                                  shard_float16_params_this_group):
                            gbuf_index, dtype, bucket_index = self.model_param_gbuf_map[model_param]
                            gbuf_range = self.gbuf_ranges[gbuf_index][dtype][bucket_index]
                            param_range = gbuf_range["param_map"][model_param]["param"]

                            shard_model_param.assign_value(model_param.detach().view(-1)[
                                                           param_range.start: param_range.end
                                                           ])
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