# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib

from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_virtual_pipeline_model_parallel_world_size,
    get_pipeline_model_parallel_world_size,
)
from mindspeed.core.weight_grad_store import WeightGradStore
from mindspeed.core.pipeline_parallel.flexible_schedules import _communicate


class InterleaveScheduleInfo():
    """
    This class manages scheduling and synchronization for interleaved pipeline parallelism. 
    It facilitates tasks such as determining the position of microbatches in a pipeline, 
    managing gradient synchronization, and handling pipeline stages.
    """
    pipeline_parallel_size: int = -1
    num_model_chunks:int = -1
    total_num_microbatches: int = -1
    no_sync_context = None
    no_sync_func = None

    @classmethod
    def disable_grad_sync(cls):
        """Disable asynchronous grad reductions"""
        if cls.no_sync_context is None:
            cls.no_sync_context = cls.no_sync_func()
            cls.no_sync_context.__enter__()
    
    @classmethod
    def enable_grad_sync(cls):
        """Enable asynchronous grad reductions"""
        if cls.no_sync_context is not None:
            cls.no_sync_context.__exit__(None, None, None)
            cls.no_sync_context = None

    @classmethod
    def init_info(cls, pipeline_parallel_size, num_model_chunks, total_num_microbatches, config):
        cls.pipeline_parallel_size = pipeline_parallel_size
        cls.num_model_chunks = num_model_chunks
        cls.total_num_microbatches = total_num_microbatches
        # Disable async grad reductions
        cls.no_sync_func = config.no_sync_func
        if isinstance(cls.no_sync_func, list):

            def multi_no_sync():
                stack = contextlib.ExitStack()
                for model_chunk_no_sync_func in config.no_sync_func:
                    stack.enter_context(model_chunk_no_sync_func())
                return stack

            cls.no_sync_func = multi_no_sync
        if cls.no_sync_func is None:
            cls.no_sync_func = contextlib.nullcontext
        cls.no_sync_context = None

    @classmethod
    def get_model_chunk_id(cls, microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (cls.pipeline_parallel_size * cls.num_model_chunks)
        model_chunk_id = microbatch_id_in_group // cls.pipeline_parallel_size
        if not forward:
            model_chunk_id = cls.num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    @classmethod
    def get_microbatch_id_in_model_chunk(cls, iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        iteration_group_id = iteration_id // (cls.pipeline_parallel_size * cls.num_model_chunks)
        microbatch_id_in_model_chunk = (iteration_group_id * cls.pipeline_parallel_size) + (
            iteration_id % cls.pipeline_parallel_size
        )
        return microbatch_id_in_model_chunk

    @classmethod
    def is_first_microbatch_for_model_chunk(cls, microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_group_size = cls.pipeline_parallel_size * cls.num_model_chunks
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % cls.pipeline_parallel_size == 0
        else:
            return False

    @classmethod
    def is_last_microbatch_for_model_chunk(cls, microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = cls.pipeline_parallel_size * cls.num_model_chunks
        num_microbatch_groups = cls.total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % cls.pipeline_parallel_size == cls.pipeline_parallel_size - 1
        else:
            return False

    @classmethod
    def microbatch_id_is_pipeline_first_stage(cls, microbatch_id, ignore_virtual=False, forward=True):
        """Return True if microbatch_id is in the first pipeline model-parallel stage, False otherwise."""
        model_chunk_id = cls.get_model_chunk_id(microbatch_id, forward=forward)
        if not ignore_virtual:
            if (
                    get_virtual_pipeline_model_parallel_world_size() is not None
                    and model_chunk_id != 0
            ):
                return False
        return get_pipeline_model_parallel_rank() == 0

    @classmethod
    def recv_forward_for_microbatch_id(cls, tensor_shape, config, group, microbatch_id):
        """Receive the input tensor for the forward pass of a given microbatch."""
        if tensor_shape is None or cls.microbatch_id_is_pipeline_first_stage(microbatch_id):
            input_tensor = None
            wait_handle = None
        else:
            input_tensor, _, wait_handle = _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=tensor_shape,
                config=config,
                group=group,
                wait_on_reqs=False
            )
        return input_tensor, wait_handle

    @classmethod
    def microbatch_id_is_pipeline_last_stage(cls, microbatch_id, ignore_virtual=False, forward=True):
        """Return True if microbatch_id is in the last pipeline model-parallel stage, False otherwise."""
        model_chunk_id = cls.get_model_chunk_id(microbatch_id, forward=forward)
        if not ignore_virtual:
            virtual_pipeline_model_parallel_world_size = get_virtual_pipeline_model_parallel_world_size()
            if (virtual_pipeline_model_parallel_world_size is not None
                    and model_chunk_id != virtual_pipeline_model_parallel_world_size - 1
            ):
                return False
        return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)
    
    @classmethod
    def recv_backward_for_microbatch_id(cls, tensor_shape, config, group, microbatch_id):
        """Receive the output gradient tensor for the backward pass of a given microbatch."""
        if tensor_shape is None or cls.microbatch_id_is_pipeline_last_stage(microbatch_id, forward=False):
            output_tensor_grad = None
            wait_handle = None
        else:
            _, output_tensor_grad, wait_handle = _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                tensor_shape=tensor_shape,
                config=config,
                group=group,
                wait_on_reqs=False
            )
        return output_tensor_grad, wait_handle


class NanoPipe():
    """
    This class controls backward propagation and gradient communication in pipelined training by decoupling weight gradient computation from the main pipeline schedule.    
    """
    num_fwd = -1
    num_dx = -1
    use_nano_swap = False
    nano_flag = []

    @classmethod
    def init_decouple(cls, pipeline_parallel_size, num_model_chunks, total_num_microbatches, num_warmup_microbatches, use_nanopipe_swap):
        num_fwd = min((pipeline_parallel_size - 1) * 2 + (num_model_chunks - 1) * pipeline_parallel_size, total_num_microbatches)
        cls.num_dx = num_fwd - num_warmup_microbatches
        if num_model_chunks == 1: 
            # When the number of virtual pipeline size is 1, set overlap_chunks_num to 1.
            overlap_chunks_num = 1
        else:
            overlap_chunks_num = (cls.num_dx + pipeline_parallel_size - 1) // pipeline_parallel_size 
        cls.use_nano_swap = use_nanopipe_swap
        
        cls.nano_flag = [True] * num_model_chunks
        for i in range(overlap_chunks_num):
            cls.nano_flag[-i - 1] = False
        cls.use_nano_swap = use_nanopipe_swap

    @classmethod
    def backward_step_helper_warper(cls, backward_step_helper, backward_k):
        if backward_k < cls.num_dx:
            WeightGradStore.start_decouple()
        WeightGradStore.resize_ori_storage(cls.use_nano_swap)
        input_tensor_grad = backward_step_helper(backward_k)
        if WeightGradStore.is_decoupleBlock:
            WeightGradStore.flush()
        if backward_k == cls.num_dx - 1:
            WeightGradStore.end_decouple()
        return input_tensor_grad

    @classmethod
    def nanopipe_cooldown(cls, synchronized_model_chunks, config, model, pipeline_parallel_size):
        if cls.nano_flag[0] and 0 not in synchronized_model_chunks:
            config.grad_sync_func[0](model[0].parameters())
            synchronized_model_chunks.add(0)
        overlap_arg = [pipeline_parallel_size, cls.nano_flag, synchronized_model_chunks, config.grad_sync_func, model]
        WeightGradStore.pop(overlap_arg)

    @classmethod
    def nanopipe_grad_sync(cls, grad_sync_chunk_id, synchronized_model_chunks, config, model):
        if cls.nano_flag[grad_sync_chunk_id]:
            InterleaveScheduleInfo.enable_grad_sync()
            config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
            synchronized_model_chunks.add(grad_sync_chunk_id)

    @classmethod
    def nanopipe_swap_tensors(cls):
        WeightGradStore.swap_tensors()
