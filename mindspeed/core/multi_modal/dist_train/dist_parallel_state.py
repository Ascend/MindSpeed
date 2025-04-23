# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""
import os
import warnings
from datetime import timedelta
from functools import partial, wraps
from itertools import cycle
from typing import Callable, List, Optional, Sequence

import torch
from torch._C._distributed_c10d import ProcessGroup
from megatron.core import mpu
from megatron.training import get_args
from megatron.core.utils import GlobalMemoryBuffer, is_torch_min_version
from megatron.core.parallel_state import RankGenerator, create_hierarchical_parallel_groups, default_embedding_ranks, \
    default_position_embedding_ranks
from mindspeed.core.multi_modal.dist_train.dist_train_config import get_dist_model_config, get_all_config_size, \
    get_all_config

# Current subworld, adapts to the situation when different model shares one rank
_CUR_SUB_WORLD = None
ALL_SUB_WORLD = {}

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra-, pipeline, and expert) that the current rank belongs to.
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# tensor model parallel group and data parallel group combined
# used for fp8 and moe training
_TENSOR_AND_DATA_PARALLEL_GROUP = None

### Expert-related parallel states
# Naming convention:
# _EXPERT prefix in group name means it's used for expert layer in MoE models.
# _EXPERT_MODEL denotes expert parallelism which splits number of experts across the group.
# _EXPERT_TENSOR denotes tensor parallelism of expert which splits tensor across the group.
# _EXPERT_DATA denotes data parallelism of expert which replicates weight across the group.

# Expert model parallel group that current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
# Expert tensor parallel group that current rank belongs to.
_EXPERT_TENSOR_PARALLEL_GROUP = None
# Expert tensor and model combined parallel group
_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = None
# Expert tensor, model, pipeline combined parallel group
_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = None
# Expert data parallel group
_EXPERT_DATA_PARALLEL_GROUP = None
_EXPERT_DATA_PARALLEL_GROUP_GLOO = None
# Parallel state values changed on the fly
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None
_MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_TENSOR_PARALLEL_RANK = None
### End of expert related parallel states

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

_PIPELINE_MODEL_PARALLEL_DECODER_START = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_DATA_PARALLEL_WORLD_SIZE = None
_MPU_DATA_PARALLEL_RANK = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each tensor model parallel group to ease calculation of
# the first local rank in the tensor model parallel group
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each model parallel group to ease calculation of
# the first local rank in the model parallel group
_MODEL_PARALLEL_GLOBAL_RANKS = None

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS = None
# Hierarchical context parallel groups
_HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = []

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# Partial Data parallel group information with context parallel combined.
_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = None
_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = None

# combined parallel group of TP and CP
_TENSOR_AND_CONTEXT_PARALLEL_GROUP = None

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

# inner data parallel group
_INNER_DATA_PARALLEL_GROUP = None
# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None

# MOE logging
_MOE_LAYER_WISE_LOGGING_TRACKER = {}


class DetachedSubWorld:
    def __init__(self, name: str, start_rank, ranks: list):
        self.name = name
        self.ranks = ranks
        self.start_rank = start_rank

        # Intra-layer model parallel group that the current rank belongs to.
        self.tensor_model_parallel_group = None
        # Inter-layer model parallel group that the current rank belongs to.
        self.pipeline_model_parallel_group = None
        # Model parallel group (both intra- and pipeline) that the current rank belongs to.
        self.model_parallel_group = None
        # Embedding group.
        self.embedding_group = None
        # Position embedding group.
        self.position_embedding_group = None
        # Data parallel group that the current rank belongs to.
        self.data_parallel_group = None
        self.data_parallel_group_gloo = None
        # tensor model parallel group and data parallel group combined
        # used for fp8 and moe training
        self.tensor_and_data_parallel_group = None

        ### Expert-related parallel states
        # Naming convention:
        # _EXPERT prefix in group name means it's used for expert layer in MoE models.
        # _EXPERT_MODEL denotes expert parallelism which splits number of experts across the group.
        # _EXPERT_TENSOR denotes tensor parallelism of expert which splits tensor across the group.
        # _EXPERT_DATA denotes data parallelism of expert which replicates weight across the group.

        # Expert model parallel group that current rank belongs to.
        self.expert_model_parallel_group = None
        # Expert tensor parallel group that current rank belongs to.
        self.expert_tensor_parallel_group = None
        # Expert tensor and model combined parallel group
        self.expert_tensor_and_model_parallel_group = None
        # Expert tensor, model, pipeline combined parallel group
        self.expert_tensor_model_pipeline_parallel_group = None
        # Expert data parallel group
        self.expert_data_parallel_group = None
        self.expert_data_parallel_group_gloo = None
        # Parallel state values changed on the fly
        self.mpu_expert_model_parallel_world_size = None
        self.mpu_expert_model_parallel_rank = None
        self.mpu_expert_tensor_parallel_world_size = None
        self.mpu_expert_tensor_parallel_rank = None
        ### End of expert related parallel states

        self.virtual_pipeline_model_parallel_rank = None
        self.virtual_pipeline_model_parallel_world_size = None
        self.pipeline_model_parallel_split_rank = None

        self.pipeline_model_parallel_decoder_start = None

        # These values enable us to change the mpu sizes on the fly.
        self.mpu_tensor_model_parallel_world_size = None
        self.mpu_pipeline_model_parallel_world_size = None
        self.mpu_data_parallel_world_size = None
        self.mpu_data_parallel_rank = None
        self.mpu_tensor_model_parallel_rank = None
        self.mpu_pipeline_model_parallel_rank = None

        # A list of ranks that have a copy of the embedding.
        self.embedding_global_ranks = None

        # A list of ranks that have a copy of the position embedding.
        self.position_embedding_global_ranks = None

        # A list of global ranks for each pipeline group to ease calculation of the source
        # rank when broadcasting from the first or last pipeline stage.
        self.pipeline_global_ranks = None

        # A list of global ranks for each data parallel group to ease calculation of the source
        # rank when broadcasting weights from src to all other data parallel ranks
        self.data_parallel_global_ranks = None

        # A list of global ranks for each tensor model parallel group to ease calculation of
        # the first local rank in the tensor model parallel group
        self.tensor_model_parallel_global_ranks = None

        # A list of global ranks for each model parallel group to ease calculation of
        # the first local rank in the model parallel group
        self.model_parallel_global_ranks = None

        # Context parallel group that the current rank belongs to
        self.context_parallel_group = None
        # A list of global ranks for each context parallel group to ease calculation of the
        # destination rank when exchanging KV/dKV between context parallel_ranks
        self.context_parallel_global_ranks = None
        # Hierarchical context parallel groups
        self.hierarchical_context_parallel_groups = []

        # Data parallel group information with context parallel combined.
        self.data_parallel_group_with_cp = None
        self.data_parallel_group_with_cp_gloo = None
        self.data_parallel_global_ranks_with_cp = None

        # Partial data parallel group information with context parallel combined.
        self.intra_partial_data_parallel_group_with_cp = None
        self.intra_partial_data_parallel_group_with_cp_gloo = None
        self.inter_partial_data_parallel_group_with_cp = None

        # combined parallel group of TP and CP
        self.tensor_and_context_parallel_group = None

        # combined parallel group of TP, DP, and CP used for fp8
        self.tensor_and_data_parallel_group_with_cp = None

        # inner data parallel group
        self.inner_data_parallel_group = None
        # Memory buffers to avoid dynamic memory allocation
        self.global_memory_buffer = None

        # MOE logging
        self.moe_layer_wise_logging_tracker = {}

    def __repr__(self):
        repr_str = ""

        print_keys = {"name": "model",
                      "pipeline_model_parallel_group": "PP_RANKS",
                      "tensor_model_parallel_group": "TP_RANKS",
                      "data_parallel_group": "DP_RANKS",
                      "context_parallel_group": "CP_RANKS",
                      "tensor_and_data_parallel_group": "TP_DP_RANKS",
                      "tensor_and_expert_parallel_group": "TP_EP_RANKS"}

        for name, value in vars(self).items():
            if name not in print_keys:
                continue
            else:
                name = print_keys[name]

            repr_str += f"{name}="
            if isinstance(value, range):
                repr_str += f"{list(value)},"
            elif isinstance(value, ProcessGroup):
                if value is not None:
                    repr_str += f"{torch.distributed.get_process_group_ranks(value)},"
                else:
                    repr_str += f"{value},"
            else:
                repr_str += f"{value},"

        return repr_str


def reset_global_group_and_ranks():
    # create an empty subworld, then use its members' default value to reset global group and ranks
    empty_subworld = DetachedSubWorld("empty_subworld", 0, [0])
    set_global_group_and_ranks_by_subworld(empty_subworld)


def set_global_group_and_ranks_by_subworld(subworld: DetachedSubWorld):
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _MODEL_PARALLEL_GROUP
    global _EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GROUP
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _EXPERT_MODEL_PARALLEL_GROUP
    global _EXPERT_TENSOR_PARALLEL_GROUP
    global _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    global _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP
    global _EXPERT_DATA_PARALLEL_GROUP
    global _EXPERT_DATA_PARALLEL_GROUP_GLOO
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    global _PIPELINE_MODEL_PARALLEL_DECODER_START
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    global _MPU_DATA_PARALLEL_WORLD_SIZE
    global _MPU_DATA_PARALLEL_RANK
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    global _EMBEDDING_GLOBAL_RANKS
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    global _PIPELINE_GLOBAL_RANKS
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    global _MODEL_PARALLEL_GLOBAL_RANKS
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    global _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    global _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    global _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
    global _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    global _INNER_DATA_PARALLEL_GROUP
    global _GLOBAL_MEMORY_BUFFER
    global _MOE_LAYER_WISE_LOGGING_TRACKER
    # Intra-layer model parallel group that the current rank belongs to.
    _TENSOR_MODEL_PARALLEL_GROUP = subworld.tensor_model_parallel_group
    # Inter-layer model parallel group that the current rank belongs to.
    _PIPELINE_MODEL_PARALLEL_GROUP = subworld.pipeline_model_parallel_group
    # Model parallel group (both intra- and pipeline) that the current rank belongs to.
    _MODEL_PARALLEL_GROUP = subworld.model_parallel_group
    # Embedding group.
    _EMBEDDING_GROUP = subworld.embedding_group
    # Position embedding group.
    _POSITION_EMBEDDING_GROUP = subworld.position_embedding_group
    # Data parallel group that the current rank belongs to.
    _DATA_PARALLEL_GROUP = subworld.data_parallel_group
    _DATA_PARALLEL_GROUP_GLOO = subworld.data_parallel_group_gloo
    # tensor model parallel group and data parallel group combined
    # used for fp8 and moe training
    _TENSOR_AND_DATA_PARALLEL_GROUP = subworld.tensor_and_data_parallel_group

    ### Expert-related parallel states
    # Naming convention:
    # _EXPERT prefix in group name means it's used for expert layer in MoE models.
    # _EXPERT_MODEL denotes expert parallelism which splits number of experts across the group.
    # _EXPERT_TENSOR denotes tensor parallelism of expert which splits tensor across the group.
    # _EXPERT_DATA denotes data parallelism of expert which replicates weight across the group.

    # Expert model parallel group that current rank belongs to.
    _EXPERT_MODEL_PARALLEL_GROUP = subworld.expert_model_parallel_group
    # Expert tensor parallel group that current rank belongs to.
    _EXPERT_TENSOR_PARALLEL_GROUP = subworld.expert_tensor_parallel_group
    # Expert tensor and model combined parallel group
    _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = subworld.expert_tensor_and_model_parallel_group
    # Expert tensor, model, pipeline combined parallel group
    _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = subworld.expert_tensor_model_pipeline_parallel_group
    # Expert data parallel group
    _EXPERT_DATA_PARALLEL_GROUP = subworld.expert_data_parallel_group
    _EXPERT_DATA_PARALLEL_GROUP_GLOO = subworld.expert_data_parallel_group_gloo
    # Parallel state values changed on the fly
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = subworld.mpu_expert_model_parallel_world_size
    _MPU_EXPERT_MODEL_PARALLEL_RANK = subworld.mpu_expert_model_parallel_rank
    _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE = subworld.mpu_expert_tensor_parallel_world_size
    _MPU_EXPERT_TENSOR_PARALLEL_RANK = subworld.mpu_expert_tensor_parallel_rank
    ### End of expert related parallel states

    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = subworld.virtual_pipeline_model_parallel_rank
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = subworld.virtual_pipeline_model_parallel_world_size
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = subworld.pipeline_model_parallel_split_rank

    _PIPELINE_MODEL_PARALLEL_DECODER_START = subworld.pipeline_model_parallel_decoder_start

    # These values enable us to change the mpu sizes on the fly.
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = subworld.mpu_tensor_model_parallel_world_size
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = subworld.mpu_pipeline_model_parallel_world_size
    _MPU_DATA_PARALLEL_WORLD_SIZE = subworld.mpu_data_parallel_world_size
    _MPU_DATA_PARALLEL_RANK = subworld.mpu_data_parallel_rank
    _MPU_TENSOR_MODEL_PARALLEL_RANK = subworld.mpu_tensor_model_parallel_rank
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = subworld.mpu_pipeline_model_parallel_rank

    # A list of ranks that have a copy of the embedding.
    _EMBEDDING_GLOBAL_RANKS = subworld.embedding_global_ranks

    # A list of ranks that have a copy of the position embedding.
    _POSITION_EMBEDDING_GLOBAL_RANKS = subworld.position_embedding_global_ranks

    # A list of global ranks for each pipeline group to ease calculation of the source
    # rank when broadcasting from the first or last pipeline stage.
    _PIPELINE_GLOBAL_RANKS = subworld.pipeline_global_ranks

    # A list of global ranks for each data parallel group to ease calculation of the source
    # rank when broadcasting weights from src to all other data parallel ranks
    _DATA_PARALLEL_GLOBAL_RANKS = subworld.data_parallel_global_ranks

    # A list of global ranks for each tensor model parallel group to ease calculation of
    # the first local rank in the tensor model parallel group
    _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = subworld.tensor_model_parallel_global_ranks

    # A list of global ranks for each model parallel group to ease calculation of
    # the first local rank in the model parallel group
    _MODEL_PARALLEL_GLOBAL_RANKS = subworld.model_parallel_global_ranks

    # Context parallel group that the current rank belongs to
    _CONTEXT_PARALLEL_GROUP = subworld.context_parallel_group
    # A list of global ranks for each context parallel group to ease calculation of the
    # destination rank when exchanging KV/dKV between context parallel_ranks
    _CONTEXT_PARALLEL_GLOBAL_RANKS = subworld.context_parallel_global_ranks
    # Hierarchical context parallel groups
    _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = subworld.hierarchical_context_parallel_groups

    # Data parallel group information with context parallel combined.
    _DATA_PARALLEL_GROUP_WITH_CP = subworld.data_parallel_group_with_cp
    _DATA_PARALLEL_GROUP_WITH_CP_GLOO = subworld.data_parallel_group_with_cp_gloo
    _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = subworld.data_parallel_global_ranks_with_cp

    # Partial data parallel group information with context parallel combined.
    _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = subworld.intra_partial_data_parallel_group_with_cp
    _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = subworld.intra_partial_data_parallel_group_with_cp_gloo
    _INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = subworld.inter_partial_data_parallel_group_with_cp

    # combined parallel group of TP and CP
    _TENSOR_AND_CONTEXT_PARALLEL_GROUP = subworld.tensor_and_context_parallel_group

    # combined parallel group of TP, DP, and CP used for fp8
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = subworld.tensor_and_data_parallel_group_with_cp

    # inner data parallel group
    _INNER_DATA_PARALLEL_GROUP = subworld.inner_data_parallel_group

    # Memory buffers to avoid dynamic memory allocation
    _GLOBAL_MEMORY_BUFFER = subworld.global_memory_buffer

    # MOE logging
    _MOE_LAYER_WISE_LOGGING_TRACKER = subworld.moe_layer_wise_logging_tracker


def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Args:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations

    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        if 'net_name' in nccl_comm_cfgs[pg_name]:
            nccl_options.config.net_name = nccl_comm_cfgs[pg_name].get('net_name')
            # verify net_name value
            if nccl_options.config.net_name.lower() not in ['ib', 'socket']:
                raise RuntimeError(
                    f"net_name ({nccl_options.config.net_name}) is not supported."
                    f"Accepted values: 'IB' or 'socket'."
                )
        return nccl_options
    else:
        return None


def is_last_rank():
    global _CUR_SUB_WORLD
    rank = torch.distributed.get_rank()
    if _CUR_SUB_WORLD is None:
        raise RuntimeError('_CUR_SUB_WORLD should not be None')
    if rank == _CUR_SUB_WORLD.ranks[-1]:
        return True
    return False


def _initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_split_rank: Optional[int] = None,
        use_sharp: bool = False,
        context_parallel_size: int = 1,
        hierarchical_context_parallel_sizes: Optional[List[int]] = None,
        expert_model_parallel_size: int = 1,
        num_distributed_optimizer_instances: int = 1,
        expert_tensor_parallel_size: Optional[int] = None,
        nccl_communicator_config_path: Optional[str] = None,
        distributed_timeout_minutes: int = 30,
        order: str = "tp-cp-ep-dp-pp",
        encoder_tensor_model_parallel_size: int = 0,
        encoder_pipeline_model_parallel_size: Optional[int] = 0,
        get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
        get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
        subworld: DetachedSubWorld = None
):
    if encoder_pipeline_model_parallel_size is None:
        encoder_pipeline_model_parallel_size = 0

    if encoder_tensor_model_parallel_size == 0 and encoder_pipeline_model_parallel_size > 0:
        encoder_tensor_model_parallel_size = tensor_model_parallel_size

    if get_embedding_ranks is None:
        get_embedding_ranks = partial(
            default_embedding_ranks, split_rank=pipeline_model_parallel_split_rank
        )

    if get_position_embedding_ranks is None:
        get_position_embedding_ranks = partial(
            default_position_embedding_ranks, split_rank=pipeline_model_parallel_split_rank
        )

    if encoder_pipeline_model_parallel_size > 0:
        subworld.pipeline_model_parallel_decoder_start = encoder_pipeline_model_parallel_size
    # Get world size and rank. Ensure some consistencies.
    tp_ranks = []
    pp_ranks = []
    if subworld is None:
        return pp_ranks, tp_ranks

    if not torch.distributed.is_initialized():
        raise RuntimeError('Distributed is not initialized.')
    world_size: int = torch.distributed.get_world_size()
    sub_world_size = len(subworld.ranks)
    if sub_world_size > world_size:
        raise RuntimeError(f"world_size ({world_size}) is less than sub_world_size ({sub_world_size})")
    world_size = sub_world_size
    reset_global_group_and_ranks()

    def adjust_rank(ranks_: Sequence):
        for i_, _ in enumerate(ranks_):
            ranks_[i_] += subworld.start_rank
        return ranks_

    if encoder_tensor_model_parallel_size > 0:
        assert (
                encoder_tensor_model_parallel_size <= tensor_model_parallel_size
        ), "We do not support encoders with more TP than the decoder."

    encoder_model_size = (
            encoder_tensor_model_parallel_size
            * encoder_pipeline_model_parallel_size
            * context_parallel_size
    )
    decoder_model_size = (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )
    total_model_size = encoder_model_size + decoder_model_size

    if world_size % total_model_size != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by {total_model_size}")

    data_parallel_size: int = world_size // total_model_size

    encoder_world_size = encoder_model_size * data_parallel_size
    decoder_world_size = decoder_model_size * data_parallel_size

    assert (
            encoder_world_size + decoder_world_size == world_size
    ), f"{encoder_world_size=} + {decoder_world_size=} != {world_size=}"

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 1:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 1 with interleaved schedule"
            )
        subworld.virtual_pipeline_model_parallel_rank = 0
        subworld.virtual_pipeline_model_parallel_world_size = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        subworld.pipeline_model_parallel_split_rank = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    if encoder_world_size > 0:
        encoder_rank_generator = RankGenerator(
            tp=encoder_tensor_model_parallel_size,
            ep=1,
            dp=data_parallel_size,
            pp=encoder_pipeline_model_parallel_size,
            cp=context_parallel_size,
            order=order,
            rank_offset=0,
        )
    else:
        encoder_rank_generator = None

    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=1,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
        rank_offset=encoder_world_size,
    )

    # Build expert rank generator
    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = tensor_model_parallel_size
    expert_tensor_model_pipeline_parallel_size = (
            expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size
    )
    expert_data_parallel_size = decoder_world_size // expert_tensor_model_pipeline_parallel_size
    if decoder_world_size % expert_tensor_model_pipeline_parallel_size != 0:
        raise RuntimeError(
            f"decoder world_size ({decoder_world_size}) is not divisible by expert_tensor_model_pipeline_parallel size ({expert_tensor_model_pipeline_parallel_size})"
        )

    # TODO: support expert specific ordering
    expert_decoder_rank_generator = RankGenerator(
        tp=expert_tensor_parallel_size,
        ep=expert_model_parallel_size,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=1,
        order=order,
        rank_offset=encoder_world_size,
    )

    assert decoder_rank_generator.get_ranks("pp") == expert_decoder_rank_generator.get_ranks(
        "pp"
    ), f"Pipeline parallel groups are expected to be the same for Non-Expert and Expert part, \
    but got {decoder_rank_generator.get_ranks('pp')} and {expert_decoder_rank_generator.get_ranks('pp')}"

    def generator_wrapper(group_type, is_expert=False, **kwargs):
        """The `RankGenerator` class produces a hyper-rectangle for a given set of
        tensor, pipeline, data, expert, and context parallelism. If we have an encoder,
        in addition to the default decoder, we essentially instantiate two `RankGenerator`
        classes to construct the parallelism for each module separately, and we then have
        to stitch them together for the right groups. For now, this means pp and tp-pp."""
        if is_expert:
            d_ranks = expert_decoder_rank_generator.get_ranks(group_type, **kwargs)
        else:
            d_ranks = decoder_rank_generator.get_ranks(group_type, **kwargs)

        if encoder_rank_generator is None:
            for x in d_ranks:
                yield x
            return
        e_ranks = encoder_rank_generator.get_ranks(group_type, **kwargs)
        if group_type == 'pp':
            # Map 1 encoder tp rank to several decoder tp ranks, because
            # these won't be the same size.
            for x, y in zip(cycle(e_ranks), d_ranks):
                yield x + y
        elif group_type == 'tp-pp':
            # For this group, we can just return the concatenated
            # groups together, because their sizes are the same.
            assert len(e_ranks) == len(d_ranks)
            for x, y in zip(e_ranks, d_ranks):
                yield x + y
        else:
            for x in e_ranks:
                yield x
            for x in d_ranks:
                yield x

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    assert subworld.data_parallel_group is None, 'data parallel group is already initialized'

    for ranks in generator_wrapper('dp'):
        ranks = adjust_rank(ranks)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
        )
        group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
        if rank in ranks:
            subworld.data_parallel_group = group
            subworld.data_parallel_group_gloo = group_gloo
            subworld.data_parallel_global_ranks = ranks

    intra_partial_data_parallel_size = data_parallel_size // num_distributed_optimizer_instances
    for ranks_with_cp in generator_wrapper('dp-cp'):
        ranks_with_cp = adjust_rank(ranks_with_cp)
        group_with_cp = torch.distributed.new_group(
            ranks_with_cp, timeout=timeout, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)
        )
        group_with_cp_gloo = torch.distributed.new_group(
            ranks_with_cp, timeout=timeout, backend="gloo"
        )
        if rank in ranks_with_cp:
            subworld.data_parallel_group_with_cp = group_with_cp
            subworld.data_parallel_group_with_cp_gloo = group_with_cp_gloo
            subworld.data_parallel_global_ranks_with_cp = ranks_with_cp
        if num_distributed_optimizer_instances > 1:
            # Create groups for Partial DistOpt, one for intra-partial DP domain
            # Another for inter-partial DP domain
            for i in range(num_distributed_optimizer_instances):
                intra_partial_data_parallel_ranks_with_cp = ranks_with_cp[
                                                            (i * intra_partial_data_parallel_size): (
                                                                    (i + 1) * intra_partial_data_parallel_size
                                                            )
                                                            ]

                intra_partial_data_parallel_group_with_cp = torch.distributed.new_group(
                    intra_partial_data_parallel_ranks_with_cp,
                    timeout=timeout,
                    pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs),
                )
                intra_partial_data_parallel_group_with_cp_gloo = torch.distributed.new_group(
                    intra_partial_data_parallel_ranks_with_cp, timeout=timeout, backend="gloo"
                )

                if rank in intra_partial_data_parallel_ranks_with_cp:
                    subworld.intra_partial_data_parallel_group_with_cp = (
                        intra_partial_data_parallel_group_with_cp
                    )
                    subworld.intra_partial_data_parallel_group_with_cp_gloo = (
                        intra_partial_data_parallel_group_with_cp_gloo
                    )

            for i in range(intra_partial_data_parallel_size):
                inter_partial_data_parallel_ranks_with_cp = ranks_with_cp[
                                                            i::intra_partial_data_parallel_size
                                                            ]

                inter_partial_data_parallel_group_with_cp = torch.distributed.new_group(
                    inter_partial_data_parallel_ranks_with_cp,
                    timeout=timeout,
                    pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs),
                )

                if rank in inter_partial_data_parallel_ranks_with_cp:
                    subworld.inter_partial_data_parallel_group_with_cp = (
                        inter_partial_data_parallel_group_with_cp
                    )
        else:
            subworld.intra_partial_data_parallel_group_with_cp = subworld.data_parallel_group_with_cp
            subworld.intra_partial_data_parallel_group_with_cp_gloo = subworld.data_parallel_group_with_cp_gloo

    # Apply SHARP to DP process groups
    if use_sharp:
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # Build the context-parallel groups.
    assert subworld.context_parallel_group is None, 'context parallel group is already initialized'
    for ranks in generator_wrapper('cp'):
        ranks = adjust_rank(ranks)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            subworld.context_parallel_group = group
            subworld.context_parallel_global_ranks = ranks
        if hierarchical_context_parallel_sizes:
            subworld.hierarchical_context_parallel_groups += create_hierarchical_parallel_groups(
                rank,
                ranks,
                context_parallel_size,
                hierarchical_context_parallel_sizes,
                get_nccl_options('cp', nccl_comm_cfgs),
            )
    # Build the model-parallel groups.
    assert subworld.model_parallel_group is None, 'model parallel group is already initialized'
    for ranks in generator_wrapper('tp-pp'):
        ranks = adjust_rank(ranks)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in ranks:
            subworld.model_parallel_group = group
            subworld.model_parallel_global_ranks = ranks

    # Build the tensor model-parallel groups.
    assert subworld.tensor_model_parallel_group is None, 'tensor model parallel group is already initialized'
    for ranks in generator_wrapper('tp'):
        ranks = adjust_rank(ranks)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
        )
        if rank in ranks:
            subworld.tensor_model_parallel_group = group
            subworld.model_parallel_global_ranks = ranks

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    assert subworld.pipeline_model_parallel_group is None, 'pipeline model parallel group is already initialized'
    assert subworld.embedding_group is None, 'embedding group is already initialized'
    assert subworld.position_embedding_group is None, 'position embedding group is already initialized'
    for ranks in generator_wrapper('pp'):
        ranks = adjust_rank(ranks)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs)
        )
        pp_ranks.append(list(ranks))
        if rank in ranks:
            if subworld.pipeline_model_parallel_group is None:
                subworld.pipeline_model_parallel_group = group
                subworld.pipeline_global_ranks = ranks
            elif isinstance(subworld.pipeline_global_ranks[0], list):
                subworld.pipeline_model_parallel_group.append(group)
                subworld.pipeline_global_ranks.append(ranks)
            else:
                subworld.pipeline_model_parallel_group = [subworld.pipeline_model_parallel_group, group]
                subworld.pipeline_global_ranks = [subworld.pipeline_global_ranks, ranks]

        embedding_ranks = get_embedding_ranks(ranks)
        group = torch.distributed.new_group(
            embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
        )
        if rank in embedding_ranks:
            subworld.embedding_group = group
            subworld.embedding_global_ranks = embedding_ranks

        position_embedding_ranks = get_position_embedding_ranks(ranks)
        group = torch.distributed.new_group(
            position_embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options('embd', nccl_comm_cfgs),
        )
        if rank in position_embedding_ranks:
            subworld.position_embedding_group = group
            subworld.position_embedding_global_ranks = position_embedding_ranks

    # Build the tensor + data parallel groups.
    assert subworld.tensor_and_data_parallel_group is None, 'Tensor + data parallel group is already initialized'
    for ranks in generator_wrapper('tp-dp-cp'):
        ranks = adjust_rank(ranks)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            subworld.tensor_and_data_parallel_group_with_cp = group
    for ranks in generator_wrapper('tp-dp'):
        ranks = adjust_rank(ranks)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)
        )
        tp_ranks.append(list(ranks))
        if rank in ranks:
            subworld.tensor_and_data_parallel_group = group

    assert subworld.tensor_and_context_parallel_group is None, 'Tensor + context parallel group is already initialized'
    for ranks in generator_wrapper('tp-cp'):
        ranks = adjust_rank(ranks)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            subworld.tensor_and_context_parallel_group = group

    ### Expert-related parallel groups initialization
    # Build the expert model parallel group
    for ranks in generator_wrapper('ep', is_expert=True):
        ranks = adjust_rank(ranks)
        group = torch.distributed.new_group(
            ranks, pg_options=get_nccl_options('exp', nccl_comm_cfgs)
        )
        if rank in ranks:
            subworld.expert_model_parallel_group = group

    # Build the expert tensor parallel group
    for ranks in generator_wrapper('tp', is_expert=True):
        ranks = adjust_rank(ranks)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
        )
        if rank in ranks:
            subworld.expert_tensor_parallel_group = group

    # Build the tensor + expert parallel groups
    for ranks in generator_wrapper('tp-ep', is_expert=True):
        ranks = adjust_rank(ranks)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)
        )
        if rank in ranks:
            subworld.expert_tensor_and_model_parallel_group = group

    # Build the expert+tensor+pipeline parallel groups
    assert subworld.expert_tensor_model_pipeline_parallel_group is None, 'model and expert parallel group is already initialized'
    for ranks in generator_wrapper('tp-ep-pp', is_expert=True):
        ranks = adjust_rank(ranks)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in ranks:
            subworld.expert_tensor_model_pipeline_parallel_group = group

    # Build the expert data parallel group
    assert subworld.expert_data_parallel_group is None, 'Expert data group is already initialized'
    assert subworld.expert_data_parallel_group_gloo is None, 'Expert data group-gloo is already initialized'
    for ranks in generator_wrapper('dp', is_expert=True):
        ranks = adjust_rank(ranks)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
        )
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")
        if rank in ranks:
            subworld.expert_data_parallel_group = group
            subworld.expert_data_parallel_group_gloo = group_gloo

    if any(cfg.main_dp for cfg in get_all_config().values()):
        from mindspeed.core.multi_modal.dist_train.utils import get_global_data_parallel_size
        if subworld.inner_data_parallel_group is not None:
            raise RuntimeError('inner dp model parallel group is already initialized')
        if get_global_data_parallel_size() > data_parallel_size:
            raise RuntimeError(f'global dp size ({get_global_data_parallel_size()}) should smaller than or equals to '
                               f'subworld dp size ({data_parallel_size})')
        inner_dp_size = data_parallel_size // get_global_data_parallel_size()
        for i in range(world_size // inner_dp_size):
            start_rank = i * inner_dp_size
            end_rank = (i + 1) * inner_dp_size
            ranks = adjust_rank(list(range(start_rank, end_rank)))
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('inner_dp', nccl_comm_cfgs)
            )
            if rank in ranks:
                subworld.inner_data_parallel_group = group
    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer(subworld=subworld)

    # append to all sub world list
    global ALL_SUB_WORLD
    if rank in subworld.ranks:
        reset_global_group_and_ranks()
        set_global_group_and_ranks_by_subworld(subworld=subworld)
        ALL_SUB_WORLD[subworld.name] = subworld
        print(f"rank={rank},{subworld}")
    return pp_ranks, tp_ranks


def initialize_model_parallel(*args, **kwargs) -> None:
    global _CUR_SUB_WORLD, ALL_SUB_WORLD
    _CUR_SUB_WORLD = None
    ALL_SUB_WORLD = {}
    world_size: int = torch.distributed.get_world_size()
    all_cfg = []
    all_pp_and_tp_ranks = {}

    # 初始化并行组
    dist_all_world_size = 0
    for i in range(get_all_config_size()):
        cfg = get_dist_model_config(global_index=i)
        dist_all_world_size += cfg.world_size
        subworld = DetachedSubWorld(cfg.name, cfg.start_rank,
                                    list(range(cfg.start_rank, cfg.start_rank + cfg.world_size)))
        pp_ranks, tp_ranks = _initialize_model_parallel(cfg.tensor_model_parallel_size, cfg.pipeline_model_parallel_size,
                                                        context_parallel_size=cfg.context_parallel_size,
                                                        subworld=subworld)
        all_cfg.append(cfg)
        all_pp_and_tp_ranks[cfg.model_index] = all_pp_and_tp_ranks.get(cfg.model_index, []) + [[pp_ranks, tp_ranks]]
    if world_size != dist_all_world_size:
        raise RuntimeError(f"{world_size=} should equals to {dist_all_world_size=}")

    # 生成映射关系
    from mindspeed.core.multi_modal.dist_train.dist_ranks_match import generate_model_comm_ranks, get_dst_ranks
    for i in range(len(all_pp_and_tp_ranks) - 1):
        for ranks_prev in all_pp_and_tp_ranks.get(i, []):
            for ranks_post in all_pp_and_tp_ranks.get(i + 1, []):
                comm_args = ranks_prev + ranks_post
                generate_model_comm_ranks(*comm_args)
                dst_ranks = get_dst_ranks()
                if dst_ranks is not None:
                    print(f"rank={torch.distributed.get_rank()} "
                          f"--> {dst_ranks}, prev: {list(comm_args[1])}, last: {list(comm_args[3])}")


def _set_global_memory_buffer(subworld: DetachedSubWorld):
    # Initialize subworld buffer
    if subworld.global_memory_buffer is not None:
        raise RuntimeError('subworld memory buffer is already initialized')
    subworld.global_memory_buffer = GlobalMemoryBuffer()


def _get_subworld_by_name(name=""):
    if ALL_SUB_WORLD is None:
        raise RuntimeError('all subworld is not initialized')
    return ALL_SUB_WORLD.get(name, None)


def set_subworld_by_name(name=""):
    global _CUR_SUB_WORLD
    if is_in_subworld(name):
        _CUR_SUB_WORLD = _get_subworld_by_name(name)


def is_in_subworld(name=""):
    subworld = _get_subworld_by_name(name)
    if subworld is None:
        return False
    rank = torch.distributed.get_rank()
    return rank in subworld.ranks


def is_not_use_dist_train_or_in_subworld(name=""):
    args = get_args()
    if getattr(args, "dist_train", False):
        return is_in_subworld(name)
    return True


def is_use_dist_train_and_in_subworld(name=""):
    args = get_args()
    if getattr(args, "dist_train", False):
        return is_in_subworld(name)
    return False


def get_is_pipeline_first_stage_wrapper(is_pipeline_first_stage):
    @wraps(is_pipeline_first_stage)
    def wrapper(*args, **kwargs):
        return _is_pipeline_first_stage(*args, **kwargs)
    return wrapper


def _is_pipeline_first_stage(ignore_virtual=False, is_global=True):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if is_global:
        from mindspeed.core.multi_modal.dist_train.dist_train_config import get_dist_model_name
        if _get_subworld_by_name(get_dist_model_name()) is None:
            return False

    if not ignore_virtual:
        if (
                get_virtual_pipeline_model_parallel_world_size() is not None
                and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0


def get_is_pipeline_last_stage_wrapper(is_pipeline_last_stage):
    @wraps(is_pipeline_last_stage)
    def wrapper(*args, **kwargs):
        return _is_pipeline_last_stage(*args, **kwargs)
    return wrapper


def _is_pipeline_last_stage(ignore_virtual=False, is_global=True):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if is_global:
        from mindspeed.core.multi_modal.dist_train import dist_train_config
        name = dist_train_config._RANK_NUMBER_TO_MODEL_NAME[-1]
        if _get_subworld_by_name(name) is None:
            return False

    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = (
            get_virtual_pipeline_model_parallel_world_size()
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (
                virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


def subwrold_decorator(wrap_func):
    @wraps(wrap_func)
    def wrap_the_function(*args, **kwargs):
        global _CUR_SUB_WORLD
        reset_global_group_and_ranks()
        if _CUR_SUB_WORLD is None:
            from mindspeed.core.multi_modal.dist_train.dist_train_config import get_dist_model_name
            name = get_dist_model_name()
            set_subworld_by_name(name)
        if _CUR_SUB_WORLD is not None:
            set_global_group_and_ranks_by_subworld(subworld=_CUR_SUB_WORLD)
        ret = wrap_func(*args, **kwargs)
        return ret
    return wrap_the_function


def get_tensor_model_parallel_src_rank_wrapper(get_tensor_model_parallel_src_rank):
    @wraps(get_tensor_model_parallel_src_rank)
    def wrapper():
        return _get_tensor_model_parallel_src_rank()
    return wrapper


@subwrold_decorator
def _get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank in the tensor model parallel group."""
    if _CUR_SUB_WORLD is None:
        return 0
    global_rank = (torch.distributed.get_rank() - _CUR_SUB_WORLD.start_rank)
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size + _CUR_SUB_WORLD.start_rank


@subwrold_decorator
def is_initialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is not None

@subwrold_decorator
def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
            _TENSOR_MODEL_PARALLEL_GROUP is None
            or _PIPELINE_MODEL_PARALLEL_GROUP is None
            or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


@subwrold_decorator
def get_model_parallel_group(with_expert_parallel=False):
    """Get the model-parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


@subwrold_decorator
def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
                _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


@subwrold_decorator
def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
            _PIPELINE_MODEL_PARALLEL_GROUP is not None
    ), 'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP


@subwrold_decorator
def get_data_parallel_group(with_context_parallel=False, partial_data_parallel=False):
    """Get the data-parallel group the caller rank belongs to."""
    if with_context_parallel:
        if partial_data_parallel:
            assert (
                _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP is not None
            ), 'Intra partial data parallel group is not initialized'
            return _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'data parallel group with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
        assert partial_data_parallel == False, 'Partial DP for Optimizer needs to include CP'
        return _DATA_PARALLEL_GROUP


@subwrold_decorator
def get_data_parallel_group_gloo(with_context_parallel=False, partial_data_parallel=False):
    """Get the Gloo data-parallel group the caller rank belongs to."""
    if with_context_parallel:
        if partial_data_parallel:
            assert (
                _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
            ), 'Intra partial data parallel group is not initialized'
            return _INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
        ), 'data parallel group-gloo with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, 'data parallel group-gloo is not initialized'
        assert partial_data_parallel == False, 'Partial DP for Optimizer needs to include CP'
        return _DATA_PARALLEL_GROUP_GLOO

@subwrold_decorator
def get_inter_partial_data_parallel_group():
    """Get the group spanning the different partial data-parallel groups."""
    assert (
        _INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP is not None
    ), 'Inter partial data parallel group is not initialized'
    return _INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP

@subwrold_decorator
def get_context_parallel_group(check_initialized=True):
    """Get the context parallel group the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GROUP


@subwrold_decorator
def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GLOBAL_RANKS is not None, 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GLOBAL_RANKS

@subwrold_decorator
def get_hierarchical_context_parallel_groups(check_initialized=True):
    """Get the inner ring of context parallel group the caller rank belongs to."""
    if check_initialized:
        assert _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS is not None
    return _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS

@subwrold_decorator
def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, 'embedding group is not initialized'
    return _EMBEDDING_GROUP

@subwrold_decorator
def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    if _POSITION_EMBEDDING_GROUP is None:
        raise RuntimeError('position embedding group is not initialized')
    return _POSITION_EMBEDDING_GROUP


@subwrold_decorator
def get_amax_reduction_group(with_context_parallel=False, tp_only_amax_red=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    if with_context_parallel:
        if not tp_only_amax_red:
            assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
            ), 'FP8 amax reduction group is not initialized'
            return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
        else:
            assert (
                _TENSOR_AND_CONTEXT_PARALLEL_GROUP is not None
            ), 'FP8 amax reduction group is not initialized'
            return _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    else:
        if not tp_only_amax_red:
            assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP is not None
            ), 'FP8 amax reduction group is not initialized'
            return _TENSOR_AND_DATA_PARALLEL_GROUP
        else:
            assert (
                _TENSOR_MODEL_PARALLEL_GROUP is not None
            ), 'FP8 amax reduction group is not initialized'
            return _TENSOR_MODEL_PARALLEL_GROUP


@subwrold_decorator
def get_tensor_and_data_parallel_group(with_context_parallel=False):
    """Get the tensor and data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
                _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


@subwrold_decorator
def get_tensor_and_context_parallel_group():
    """Get the tensor and context parallel group the caller rank belongs to."""
    assert (
            _TENSOR_AND_CONTEXT_PARALLEL_GROUP is not None
    ), 'tensor and context parallel group is not initialized'
    return _TENSOR_AND_CONTEXT_PARALLEL_GROUP



@subwrold_decorator
def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


@subwrold_decorator
def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline-model-parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE

    pp_group = get_pipeline_model_parallel_group()
    if isinstance(pp_group, list):
        # Implicit assumption that each PP group is the same size.
        sizes = []
        for group in _PIPELINE_GLOBAL_RANKS:
            sizes.append(len(group))
        assert all(x == sizes[0] for x in sizes)
        return torch.distributed.get_world_size(group=pp_group[0])
    else:
        return torch.distributed.get_world_size(group=pp_group)


@subwrold_decorator
def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


@subwrold_decorator
def get_pipeline_model_parallel_rank():
    """Return caller's rank for the pipeline-model-parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    rank = torch.distributed.get_rank()
    pp_group = get_pipeline_model_parallel_group()
    if isinstance(pp_group, list):
        # Assume that if the caller exist in multiple PP groups, then it has the same index.
        indices = []
        for group in _PIPELINE_GLOBAL_RANKS:
            for i, r in enumerate(group):
                if r == rank:
                    indices.append(i)
        assert all(x == indices[0] for x in indices)
        return torch.distributed.get_rank(group=pp_group[0])
    else:
        return torch.distributed.get_rank(group=pp_group)


@subwrold_decorator
def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if get_args().multimodal:
            if rank == _EMBEDDING_GLOBAL_RANKS[-1]:
                return _is_pipeline_last_stage()
            else:
                return True
        else:
            if rank == _EMBEDDING_GLOBAL_RANKS[0]:
                return _is_pipeline_first_stage()
            elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
                return _is_pipeline_last_stage()
            else:
                return True
    return False


@subwrold_decorator
def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS


@subwrold_decorator
def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


@subwrold_decorator
def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE

@subwrold_decorator
def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the model parallel group."""
    assert _MODEL_PARALLEL_GLOBAL_RANKS is not None, "Model parallel group is not initialized"
    return _MODEL_PARALLEL_GLOBAL_RANKS[0]

@subwrold_decorator
def get_data_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank in the data parallel group."""
    if with_context_parallel:
        assert (
                _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    else:
        assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS[0]


@subwrold_decorator
def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first stage in the current rank's pipeline."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    if isinstance(_PIPELINE_GLOBAL_RANKS[0], list):
        # I assume the first rank is the same for all pp groups right now.
        for rank_group in _PIPELINE_GLOBAL_RANKS:
            assert rank_group[0] == _PIPELINE_GLOBAL_RANKS[0][0]
        return _PIPELINE_GLOBAL_RANKS[0][0]
    else:
        return _PIPELINE_GLOBAL_RANKS[0]


@subwrold_decorator
def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last stage in the current rank's pipeline."""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


@subwrold_decorator
def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline, for each
    pipeline-parallel group that the rank is part of.

    If it is just part of one group, an int is returned, otherwise a list of ints.
    """
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    if isinstance(_PIPELINE_GLOBAL_RANKS[0], list):
        to_return = []
        for group in _PIPELINE_GLOBAL_RANKS:
            to_return.append(group[(rank_in_pipeline + 1) % world_size])
        return to_return
    else:
        return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


@subwrold_decorator
def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that precedes the caller in the pipeline, for each
    pipeline-parallel group that the rank is part of.

    If it is just part of one group, an int is returned, otherwise a list of ints.
    """
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    if isinstance(_PIPELINE_GLOBAL_RANKS[0], list):
        to_return = []
        for group in _PIPELINE_GLOBAL_RANKS:
            to_return.append(group[(rank_in_pipeline - 1) % world_size])
        return to_return
    else:
        return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]

@subwrold_decorator
def get_data_parallel_world_size(with_context_parallel=False, partial_data_parallel=False):
    """Return world size for the data parallel group."""
    global _MPU_DATA_PARALLEL_WORLD_SIZE
    if _MPU_DATA_PARALLEL_WORLD_SIZE is not None:
        return _MPU_DATA_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(
            group=get_data_parallel_group(
                with_context_parallel=with_context_parallel,
                partial_data_parallel=partial_data_parallel,
            )
        )
    else:
        return 0

@subwrold_decorator
def get_data_parallel_rank(with_context_parallel=False, partial_data_parallel=False):
    """Return caller's rank in the data-parallel group."""
    global _MPU_DATA_PARALLEL_RANK
    if _MPU_DATA_PARALLEL_RANK is not None:
        return _MPU_DATA_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(
            group=get_data_parallel_group(
                with_context_parallel=with_context_parallel,
                partial_data_parallel=partial_data_parallel,
            )
        )
    else:
        return 0

@subwrold_decorator
def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_context_parallel_group())
    else:
        return 0


@subwrold_decorator
def get_context_parallel_rank():
    """Return caller's rank in the context-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_context_parallel_group())
    else:
        return 0

@subwrold_decorator
def get_tensor_and_context_parallel_world_size():
    """Return world size for the tensor and context-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_tensor_and_context_parallel_group())
    else:
        return 0


@subwrold_decorator
def get_tensor_and_context_parallel_rank():
    """Return caller's rank in the joint tensor-model-parallel and context-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_tensor_and_context_parallel_group())
    else:
        return 0


### Expert-related parallel states functions
@subwrold_decorator
def get_expert_model_parallel_group(check_initialized=True):
    """Get the expert-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_MODEL_PARALLEL_GROUP is not None
        ), 'expert model parallel group is not initialized'
    return _EXPERT_MODEL_PARALLEL_GROUP

@subwrold_decorator
def get_expert_model_parallel_world_size():
    """Return world size for the expert-model-parallel group."""
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_expert_model_parallel_group())
    else:
        return 0

@subwrold_decorator
def get_expert_model_parallel_rank():
    """Return caller's rank in the expert-model-parallel group."""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_expert_model_parallel_group())
    else:
        return 0

@subwrold_decorator
def get_expert_tensor_parallel_group(check_initialized=True):
    """Get the expert-tensor-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_TENSOR_PARALLEL_GROUP is not None
        ), 'Expert tensor parallel group is not initialized'
    return _EXPERT_TENSOR_PARALLEL_GROUP

@subwrold_decorator
def get_expert_tensor_parallel_world_size():
    """Return world size for the expert tensor parallel group."""
    global _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    if _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_TENSOR_PARALLEL_WORLD_SIZE
    # Use tensor parallel group world size for backward compability otherwise
    if not _EXPERT_TENSOR_PARALLEL_GROUP:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    else:
        return torch.distributed.get_world_size(group=get_expert_tensor_parallel_group())

@subwrold_decorator
def get_expert_tensor_parallel_rank():
    """Return my rank for the expert tensor parallel group."""
    global _MPU_EXPERT_TENSOR_PARALLEL_RANK
    if _MPU_EXPERT_TENSOR_PARALLEL_RANK is not None:
        return _MPU_EXPERT_TENSOR_PARALLEL_RANK
    # Use tensor parallel group rank for backward compability otherwise
    if not _EXPERT_TENSOR_PARALLEL_GROUP:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    else:
        return torch.distributed.get_rank(group=get_expert_tensor_parallel_group())

@subwrold_decorator
def get_expert_tensor_and_model_parallel_group(check_initialized=True):
    """Get the expert-tensor and expert-model group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is not None
        ), 'Expert tensor and model parallel group is not initialized'
    return _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP

@subwrold_decorator
def get_expert_tensor_and_model_parallel_world_size():
    """Return world size for the expert model parallel group times expert tensor parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size(
            group=get_expert_tensor_and_model_parallel_group()
        )
        return world_size
    else:
        return 0

@subwrold_decorator
def get_expert_tensor_and_model_parallel_rank():
    """Return caller's rank in the joint tensor- and expert-model-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_expert_tensor_and_model_parallel_group())
    else:
        return 0

@subwrold_decorator
def get_expert_tensor_model_pipeline_parallel_group():
    """Get expert tensor-model-pipeline parallel group."""
    assert (
        _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is not None
    ), 'Expert tensor-model-pipeline parallel group is not initialized'
    return _EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP

@subwrold_decorator
def get_expert_data_parallel_group():
    """Get expert data parallel group."""
    assert _EXPERT_DATA_PARALLEL_GROUP is not None, 'Expert data parallel group is not initialized'
    return _EXPERT_DATA_PARALLEL_GROUP

@subwrold_decorator
def get_data_modulo_expert_parallel_group():
    """[Deprecated] Get expert data parallel group."""
    warnings.warn(
        "get_data_modulo_expert_parallel_group is deprecated, please use "
        "get_expert_data_parallel_group instead.",
        DeprecationWarning,
    )
    return get_expert_data_parallel_group()

@subwrold_decorator
def get_expert_data_parallel_group_gloo():
    """Get expert data parallel group-gloo."""
    assert (
        _EXPERT_DATA_PARALLEL_GROUP_GLOO is not None
    ), 'Expert data parallel group-gloo is not initialized'
    return _EXPERT_DATA_PARALLEL_GROUP_GLOO

@subwrold_decorator
def get_expert_data_parallel_rank():
    """Return caller's rank in the expert data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_expert_data_parallel_group())
    else:
        return 0

### End of expert-related functions region

@subwrold_decorator
def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    if _GLOBAL_MEMORY_BUFFER is None:
        raise RuntimeError('global memory buffer is not initialized')
    return _GLOBAL_MEMORY_BUFFER


@subwrold_decorator
def get_moe_layer_wise_logging_tracker():
    """Return the moe layer wise tracker."""
    global _MOE_LAYER_WISE_LOGGING_TRACKER
    return _MOE_LAYER_WISE_LOGGING_TRACKER


def get_global_pipeline_parallel_rank():
    """Return main data parallel size"""
    all_config = get_all_config()
    global_pp_rank = 0
    for config in all_config.values():
        if config.model_index < get_dist_model_config().model_index:
            global_pp_rank += config.pipeline_model_parallel_size
        else:
            global_pp_rank += get_pipeline_model_parallel_rank()
            break
    return global_pp_rank
