# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Expert parallel groups."""

from functools import wraps
from typing import Optional

import torch
import megatron

_EXPERT_PARALLEL_GROUP = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None


def initialize_model_parallel_decorator(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(
            tensor_model_parallel_size: int = 1,
            pipeline_model_parallel_size: int = 1,
            virtual_pipeline_model_parallel_size: Optional[int] = None,
            pipeline_model_parallel_split_rank: Optional[int] = None,
            use_sharp: bool = False,
            context_parallel_size: int = 1,
            expert_model_parallel_size: int = 1,
            nccl_communicator_config_path: Optional[str] = None,
    ):
        from megatron.utils import print_rank_0

        initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank,
            use_sharp,
            context_parallel_size,
            expert_model_parallel_size,
            nccl_communicator_config_path,
        )

        rank = torch.distributed.get_rank()
        world_size: int = torch.distributed.get_world_size()
        num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
        num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

        nccl_comm_cfgs = {}
        if nccl_communicator_config_path is not None:
            import yaml

            with open(nccl_communicator_config_path, "r") as stream:
                nccl_comm_cfgs = yaml.safe_load(stream)

        all_data_parallel_group_ranks = []
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(context_parallel_size * tensor_model_parallel_size):
                ranks = range(
                    start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
                )
                all_data_parallel_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, pg_options=megatron.core.parallel_state.get_nccl_options('dp', nccl_comm_cfgs)
                )

                if rank in ranks:
                    megatron.core.parallel_state._DATA_PARALLEL_GROUP = group
                    megatron.core.parallel_state._DATA_PARALLEL_GROUP_GLOO = None
                    megatron.core.parallel_state._DATA_PARALLEL_GLOBAL_RANKS = ranks
            for j in range(tensor_model_parallel_size):
                ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
                group_with_cp = torch.distributed.new_group(
                    ranks_with_cp, pg_options=megatron.core.parallel_state.get_nccl_options('dp_cp', nccl_comm_cfgs)
                )

                if rank in ranks_with_cp:
                    megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                    megatron.core.parallel_state._DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
                    megatron.core.parallel_state._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

        # Build expert parallel groups
        global _EXPERT_PARALLEL_GROUP
        if _EXPERT_PARALLEL_GROUP is not None:
            raise AttributeError('Expert parallel group is already initialized')

        all_ep_groups = []
        for dp_ranks in all_data_parallel_group_ranks:
            for i in range(0, len(dp_ranks), expert_model_parallel_size):
                ranks = dp_ranks[i:i + expert_model_parallel_size]
                all_ep_groups.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, pg_options=megatron.core.parallel_state.get_nccl_options('exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _EXPERT_PARALLEL_GROUP = group

        all_tp_groups = []
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            all_tp_groups.append(list(ranks))

        print_rank_0(f"all tp gourps {all_tp_groups}")
        print_rank_0(f"all ep groups {all_ep_groups}")
        print_rank_0(f"all dp groups {all_data_parallel_group_ranks}")

    return wrapper


def set_expert_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def set_expert_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_expert_parallel_rank():
    """Return my rank for the expert parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_expert_parallel_group())
    else:
        return 0


def get_expert_parallel_world_size():
    """Return world size for the expert parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_expert_parallel_group())
    else:
        return 0


def get_expert_parallel_group():
    if _EXPERT_PARALLEL_GROUP is None:
        raise AttributeError('Expert parallel group is not initialized')
    return _EXPERT_PARALLEL_GROUP


def get_expert_model_parallel_rank():
    """Return my rank for the expert parallel group"""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    if _MPU_EXPERT_MODEL_PARALLEL_RANK is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(
            group=megatron.core.parallel_state.get_tensor_and_expert_parallel_group()
        )
        res = tensor_and_expert_parallel_rank // \
              megatron.core.parallel_state.get_tensor_model_parallel_world_size()
    else:
        res = 0
    return res


def get_expert_model_parallel_world_size():
    """Return my rank for the expert parallel group"""
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=megatron.core.parallel_state.get_tensor_and_expert_parallel_group()
        )
        res = tensor_and_expert_parallel_world_size // \
              megatron.core.parallel_state.get_tensor_model_parallel_world_size()
    else:
        res = 0
    return res


def destroy_model_parallel_decorator(destroy_model_parallel):
    @wraps(destroy_model_parallel)
    def wrapper():
        destroy_model_parallel()

        global _EXPERT_PARALLEL_GROUP
        global _MPU_EXPERT_MODEL_PARALLEL_RANK
        global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
        _EXPERT_PARALLEL_GROUP = None
        _MPU_EXPERT_MODEL_PARALLEL_RANK = None
        _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None

    return wrapper
