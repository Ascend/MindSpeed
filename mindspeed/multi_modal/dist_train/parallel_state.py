# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""
import os
from datetime import timedelta
from typing import Optional
from functools import wraps
import torch
from torch._C._distributed_c10d import ProcessGroup
import megatron.core.parallel_state as ps
from megatron.core.utils import GlobalMemoryBuffer
from megatron.training import get_args
from .config.dist_train_config import get_dist_model_config, get_all_config_size, get_all_config


_CUR_SUB_WORLD = None  # Current subworld, adapts to the situation when different model shares one rank
ALL_SUB_WORLD = {}

_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
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
# Expert parallel group that the current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
_TENSOR_AND_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None

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

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS = None

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

# inner data parallel group
_INNER_DATA_PARALLEL_GROUP = None
# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None


class DetachedSubWorld:
    def __init__(self, name: str, start_rank, ranks: list):
        self.name = name
        self.ranks = ranks
        self.start_rank = start_rank

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
        # Expert parallel group that the current rank belongs to.
        self.expert_model_parallel_group = None
        self.tensor_and_expert_parallel_group = None
        self.data_modulo_expert_parallel_group = None
        self.data_modulo_expert_parallel_group_gloo = None

        self.virtual_pipeline_model_parallel_rank = None
        self.virtual_pipeline_model_parallel_world_size = None
        self.pipeline_model_parallel_split_rank = None

        # These values enable us to change the mpu sizes on the fly.
        self.mpu_tensor_model_parallel_world_size = None
        self.mpu_pipeline_model_parallel_world_size = None
        self.mpu_expert_model_parallel_world_size = None
        self.mpu_tensor_model_parallel_rank = None
        self.mpu_pipeline_model_parallel_rank = None
        self.mpu_expert_model_parallel_rank = None

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

        # Context parallel group that the current rank belongs to
        self.context_parallel_group = None
        # A list of global ranks for each context parallel group to ease calculation of the
        # destination rank when exchanging KV/dKV between context parallel_ranks
        self.context_parallel_global_ranks = None

        # Data parallel group information with context parallel combined.
        self.data_parallel_group_with_cp = None
        self.data_parallel_group_with_cp_gloo = None
        self.data_parallel_global_ranks_with_cp = None

        # combined parallel group of TP, DP, and CP used for fp8
        self.tensor_and_data_parallel_group_with_cp = None

        # inner data parallel group
        self.inner_data_parallel_group = None
        # Memory buffers to avoid dynamic memory allocation
        self.global_memory_buffer = None

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
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    global _EMBEDDING_GLOBAL_RANKS
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    global _PIPELINE_GLOBAL_RANKS
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    global _INNER_DATA_PARALLEL_GROUP
    global _GLOBAL_MEMORY_BUFFER

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
    # Expert parallel group that the current rank belongs to.
    _EXPERT_MODEL_PARALLEL_GROUP = subworld.expert_model_parallel_group
    _TENSOR_AND_EXPERT_PARALLEL_GROUP = subworld.tensor_and_expert_parallel_group
    _DATA_MODULO_EXPERT_PARALLEL_GROUP = subworld.data_modulo_expert_parallel_group
    _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = subworld.data_modulo_expert_parallel_group_gloo

    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = subworld.virtual_pipeline_model_parallel_rank
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = subworld.virtual_pipeline_model_parallel_world_size
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = subworld.pipeline_model_parallel_split_rank

    # These values enable us to change the mpu sizes on the fly.
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = subworld.mpu_tensor_model_parallel_world_size
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = subworld.mpu_pipeline_model_parallel_world_size
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = subworld.mpu_expert_model_parallel_world_size
    _MPU_TENSOR_MODEL_PARALLEL_RANK = subworld.mpu_tensor_model_parallel_rank
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = subworld.mpu_pipeline_model_parallel_rank
    _MPU_EXPERT_MODEL_PARALLEL_RANK = subworld.mpu_expert_model_parallel_rank

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

    # Context parallel group that the current rank belongs to
    _CONTEXT_PARALLEL_GROUP = subworld.context_parallel_group
    # A list of global ranks for each context parallel group to ease calculation of the
    # destination rank when exchanging KV/dKV between context parallel_ranks
    _CONTEXT_PARALLEL_GLOBAL_RANKS = subworld.context_parallel_global_ranks

    # Data parallel group information with context parallel combined.
    _DATA_PARALLEL_GROUP_WITH_CP = subworld.data_parallel_group_with_cp
    _DATA_PARALLEL_GROUP_WITH_CP_GLOO = subworld.data_parallel_group_with_cp_gloo
    _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = subworld.data_parallel_global_ranks_with_cp

    # combined parallel group of TP, DP, and CP used for fp8
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = subworld.tensor_and_data_parallel_group_with_cp

    # inner data parallel group
    _INNER_DATA_PARALLEL_GROUP = subworld.inner_data_parallel_group

    # Memory buffers to avoid dynamic memory allocation
    _GLOBAL_MEMORY_BUFFER = subworld.global_memory_buffer


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
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    subworld: DetachedSubWorld = None,
):
    """ Initialize model data parallel groups. """
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

    def adjust_rank(start_rank, end_rank):
        if subworld is not None:
            return start_rank + subworld.start_rank, end_rank + subworld.start_rank
        return start_rank, end_rank

    if (
        world_size
        % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
        != 0
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    data_parallel_size: int = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
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
        except ImportError as e:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            ) from e

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    if subworld.data_parallel_group is not None:
        raise RuntimeError('data parallel group is already initialized')
    all_data_parallel_group_ranks_with_cp = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        start_rank, end_rank = adjust_rank(start_rank, end_rank)
        tp_size = tensor_model_parallel_size
        for j in range(context_parallel_size * tp_size):
            ranks = list(range(start_rank + j, end_rank, context_parallel_size * tp_size))
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
            )
            group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
            if rank in ranks:
                subworld.data_parallel_group = group
                subworld.data_parallel_group_gloo = group_gloo
                subworld.data_parallel_global_ranks = ranks
        for j in range(tp_size):
            ranks_with_cp = list(range(start_rank + j, end_rank, tp_size))
            all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
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
            group=ps.get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # Build the context-parallel groups.
    if subworld.context_parallel_group is not None:
        raise RuntimeError('context parallel group is already initialized')
    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                i * num_pipeline_model_parallel_groups
                + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                i * num_pipeline_model_parallel_groups
                + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            start_rank, end_rank = adjust_rank(start_rank, end_rank)
            for k in range(tensor_model_parallel_size):
                ranks = list(range(start_rank + k, end_rank, tensor_model_parallel_size))
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=get_nccl_options('cp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    subworld.context_parallel_group = group
                    subworld.context_parallel_global_ranks = ranks

    # Build the model-parallel groups.
    if subworld.model_parallel_group is not None:
        raise RuntimeError('model parallel group is already initialized')
    for i in range(data_parallel_size * context_parallel_size):
        ranks = [
            data_parallel_group_ranks_with_cp[i]
            for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
        ]
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in ranks:
            subworld.model_parallel_group = group

    # Build the tensor model-parallel groups.
    if subworld.tensor_model_parallel_group is not None:
        raise RuntimeError('tensor model parallel group is already initialized')
    for i in range(num_tensor_model_parallel_groups):
        start_rank = i * tensor_model_parallel_size
        end_rank = (i + 1) * tensor_model_parallel_size
        start_rank, end_rank = adjust_rank(start_rank, end_rank)
        ranks = range(start_rank, end_rank)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
        )
        if rank in ranks:
            subworld.tensor_model_parallel_group = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    if subworld.pipeline_model_parallel_group is not None:
        raise RuntimeError('pipeline model parallel group is already initialized')
    if subworld.embedding_group is not None:
        raise RuntimeError('embedding group is already initialized')
    if subworld.position_embedding_group is not None:
        raise RuntimeError('position embedding group is already initialized')
    for i in range(num_pipeline_model_parallel_groups):
        start_rank = i
        end_rank = world_size
        start_rank, end_rank = adjust_rank(start_rank, end_rank)
        ranks = range(start_rank, end_rank, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs)
        )
        pp_ranks.append(list(ranks))
        if rank in ranks:
            subworld.pipeline_model_parallel_group = group
            subworld.pipeline_global_ranks = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [
                        ranks[0],
                        ranks[pipeline_model_parallel_split_rank],
                        ranks[-1],
                    ]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(
            embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
        )
        if rank in embedding_ranks:
            subworld.embedding_group = group
        if rank in ranks:
            subworld.embedding_global_ranks = embedding_ranks

        group = torch.distributed.new_group(
            position_embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options('embd', nccl_comm_cfgs),
        )
        if rank in position_embedding_ranks:
            subworld.position_embedding_group = group
        if rank in ranks:
            subworld.position_embedding_global_ranks = position_embedding_ranks

    # Build the tensor + data parallel groups.
    if subworld.tensor_and_data_parallel_group is not None:
        raise RuntimeError('Tensor + data parallel group is already initialized')
    tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * data_parallel_size * context_parallel_size
    num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
    for i in range(num_tensor_and_data_groups_with_cp):
        start_rank = i * tensor_and_data_group_size_with_cp
        end_rank = start_rank + tensor_and_data_group_size_with_cp
        start_rank, end_rank = adjust_rank(start_rank, end_rank)
        ranks = range(start_rank, end_rank)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            subworld.tensor_and_data_parallel_group_with_cp = group

        for j in range(context_parallel_size):
            ranks = []
            for k in range(data_parallel_size):
                start_rank = (
                    i * tensor_and_data_group_size_with_cp
                    + j * tensor_model_parallel_size
                    + k * tensor_model_parallel_size * context_parallel_size
                )
                end_rank = start_rank + tensor_model_parallel_size
                start_rank, end_rank = adjust_rank(start_rank, end_rank)
                ranks = ranks + list(range(start_rank, end_rank))
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)
            )
            tp_ranks.append(list(ranks))
            if rank in ranks:
                subworld.tensor_and_data_parallel_group = group

    # Build the tensor + expert parallel groups
    if subworld.expert_model_parallel_group is not None:
        raise RuntimeError('Expert parallel group is already initialized')
    if subworld.tensor_and_expert_parallel_group is not None:
        raise RuntimeError('Tensor + expert parallel group is already initialized')
    if subworld.data_modulo_expert_parallel_group is not None:
        raise RuntimeError('Data modulo expert group is already initialized')
    num_expert_groups: int = data_parallel_size * context_parallel_size // expert_model_parallel_size
    tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
    for i in range(num_tensor_and_data_groups_with_cp):
        for j in range(num_expert_groups):
            # TPxEP Group
            start_rank = i * tensor_and_data_group_size_with_cp + j * tensor_and_expert_group_size
            end_rank = i * tensor_and_data_group_size_with_cp + (j + 1) * tensor_and_expert_group_size
            start_rank, end_rank = adjust_rank(start_rank, end_rank)
            ranks = list(range(start_rank, end_rank))

            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)
            )
            if rank in ranks:
                subworld.tensor_and_expert_parallel_group = group

    tensor_and_expert_group_size_with_cp: int = tensor_model_parallel_size * expert_model_parallel_size * context_parallel_size
    num_tensor_and_expert_groups_with_cp: int = world_size // tensor_and_expert_group_size_with_cp
    for i in range(num_tensor_and_expert_groups_with_cp):
        for j in range(tensor_model_parallel_size * context_parallel_size):
            start_rank = i * tensor_and_expert_group_size_with_cp + j
            end_rank = (i + 1) * tensor_and_expert_group_size_with_cp + j
            start_rank, end_rank = adjust_rank(start_rank, end_rank)
            ranks = list(
                range(start_rank, end_rank, tensor_model_parallel_size * context_parallel_size)
            )
            group = torch.distributed.new_group(
                ranks, pg_options=get_nccl_options('exp', nccl_comm_cfgs)
            )
            if rank in ranks:
                subworld.expert_model_parallel_group = group

    tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
    num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
    tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
    for i in range(num_tensor_and_data_groups):
        start_rank = i * tensor_and_data_group_size
        end_rank = (i + 1) * tensor_and_data_group_size
        start_rank, end_rank = adjust_rank(start_rank, end_rank)
        for j in range(tensor_and_expert_group_size):
            ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
            )
            group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            if rank in ranks:
                subworld.data_modulo_expert_parallel_group = group
                subworld.data_modulo_expert_parallel_group_gloo = group_gloo

    if any(cfg.main_dp for cfg in get_all_config().values()):
        from .inner_data_parallel.utils import get_global_data_parallel_size
        if subworld.inner_data_parallel_group is not None:
            raise RuntimeError('inner dp model parallel group is already initialized')
        if get_global_data_parallel_size() > data_parallel_size:
            raise RuntimeError(f'global dp size ({get_global_data_parallel_size()}) should smaller than or equals to '
                               f'subworld dp size ({data_parallel_size})')
        inner_dp_size = data_parallel_size // get_global_data_parallel_size()
        for i in range(world_size // inner_dp_size):
            start_rank = i * inner_dp_size
            end_rank = (i + 1) * inner_dp_size
            start_rank, end_rank = adjust_rank(start_rank, end_rank)
            ranks = range(start_rank, end_rank)
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
    from .communication.dist_ranks_match import generate_model_comm_ranks, get_dst_ranks
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
        from .config.dist_train_config import get_dist_model_name
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
        from .config import dist_train_config
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
            from .config.dist_train_config import get_dist_model_name
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
def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    if _MODEL_PARALLEL_GROUP is None:
        raise RuntimeError('model parallel group is not initialized')
    return _MODEL_PARALLEL_GROUP


@subwrold_decorator
def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        if _TENSOR_MODEL_PARALLEL_GROUP is None:
            raise RuntimeError('tensor model parallel group is not initialized')
    return _TENSOR_MODEL_PARALLEL_GROUP


@subwrold_decorator
def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    if _PIPELINE_MODEL_PARALLEL_GROUP is None:
        raise RuntimeError('pipeline_model parallel group is not initialized')
    return _PIPELINE_MODEL_PARALLEL_GROUP


@subwrold_decorator
def get_data_parallel_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    if with_context_parallel:
        if _DATA_PARALLEL_GROUP_WITH_CP is None:
            raise RuntimeError('data parallel group with context parallel combined is not initialized')
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        if _DATA_PARALLEL_GROUP is None:
            raise RuntimeError('data parallel group is not initialized')
        return _DATA_PARALLEL_GROUP


@subwrold_decorator
def get_data_parallel_group_gloo(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    if with_context_parallel:
        if _DATA_PARALLEL_GROUP_WITH_CP_GLOO is None:
            raise RuntimeError('data parallel group-gloo with context parallel combined is not initialized')
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    else:
        if _DATA_PARALLEL_GROUP_GLOO is None:
            raise RuntimeError('data parallel group-gloo is not initialized')
        return _DATA_PARALLEL_GROUP_GLOO


@subwrold_decorator
def get_context_parallel_group(check_initialized=True):
    """Get the context parallel group the caller rank belongs to."""
    if check_initialized:
        if _CONTEXT_PARALLEL_GROUP is None:
            raise RuntimeError('context parallel group is not initialized')
    return _CONTEXT_PARALLEL_GROUP


@subwrold_decorator
def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    if check_initialized:
        if _CONTEXT_PARALLEL_GLOBAL_RANKS is None:
            raise RuntimeError('context parallel group is not initialized')
    return _CONTEXT_PARALLEL_GLOBAL_RANKS


@subwrold_decorator
def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    if _EMBEDDING_GROUP is None:
        raise RuntimeError('embedding group is not initialized')
    return _EMBEDDING_GROUP


@subwrold_decorator
def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    if _POSITION_EMBEDDING_GROUP is None:
        raise RuntimeError('position embedding group is not initialized')
    return _POSITION_EMBEDDING_GROUP


@subwrold_decorator
def get_amax_reduction_group(with_context_parallel=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    if with_context_parallel:
        if _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is None:
            raise RuntimeError('FP8 amax reduction group is not initialized')
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        if _TENSOR_AND_DATA_PARALLEL_GROUP is None:
            raise RuntimeError('FP8 amax reduction group is not initialized')
        return _TENSOR_AND_DATA_PARALLEL_GROUP


@subwrold_decorator
def get_tensor_and_data_parallel_group(with_context_parallel=False):
    """Get the tensor and data parallel group the caller rank belongs to."""
    if with_context_parallel:
        if _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is None:
            raise RuntimeError('tensor and data parallel group is not initialized')
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        if _TENSOR_AND_DATA_PARALLEL_GROUP is None:
            raise RuntimeError('tensor and data parallel group is not initialized')
        return _TENSOR_AND_DATA_PARALLEL_GROUP


@subwrold_decorator
def get_expert_model_parallel_group():
    if _EXPERT_MODEL_PARALLEL_GROUP is None:
        raise RuntimeError('expert model parallel group is not initialized')
    return _EXPERT_MODEL_PARALLEL_GROUP


@subwrold_decorator
def get_tensor_and_expert_parallel_group():
    if _TENSOR_AND_EXPERT_PARALLEL_GROUP is None:
        raise RuntimeError('tensor and expert parallel group is not initialized')
    return _TENSOR_AND_EXPERT_PARALLEL_GROUP


@subwrold_decorator
def get_data_modulo_expert_parallel_group():
    if _DATA_MODULO_EXPERT_PARALLEL_GROUP is None:
        raise RuntimeError('data modulo expert parallel group is not initialized')
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP


@subwrold_decorator
def get_data_modulo_expert_parallel_group_gloo():
    if _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO is None:
        raise RuntimeError('data modulo expert parallel group-gloo is not initialized')
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO


@subwrold_decorator
def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


@subwrold_decorator
def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


@subwrold_decorator
def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


@subwrold_decorator
def get_pipeline_model_parallel_rank(is_global=False):
    """Return my rank for the pipeline model parallel group."""
    if is_global:
        return get_global_pipeline_parallel_rank()
    else:
        if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
            return _MPU_PIPELINE_MODEL_PARALLEL_RANK
        return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


@subwrold_decorator
def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    embedding_ranks = get_embedding_global_ranks()
    if ignore_virtual:
        return rank in embedding_ranks
    if rank in embedding_ranks:
        if get_args().multimodal:
            if rank == embedding_ranks[-1]:
                return _is_pipeline_last_stage()
            else:
                return True
        else:
            if rank == embedding_ranks[0]:
                return _is_pipeline_first_stage()
            elif rank == embedding_ranks[-1]:
                return _is_pipeline_last_stage()
            else:
                return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    pos_embedding_ranks = get_position_embedding_ranks()
    return rank in pos_embedding_ranks


@subwrold_decorator
def get_embedding_global_ranks():
    """Return the global EmbeddingGlobalRanks object"""
    if _EMBEDDING_GLOBAL_RANKS is None:
        raise RuntimeError('global embedding global ranks is not initialized')
    return _EMBEDDING_GLOBAL_RANKS


@subwrold_decorator
def get_position_embedding_ranks():
    """Return the position embedding ranks."""
    if _POSITION_EMBEDDING_GLOBAL_RANKS is None:
        raise RuntimeError('position global embedding global ranks is not initialized')
    return _POSITION_EMBEDDING_GLOBAL_RANKS


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
def get_data_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank in the data parallel group."""
    if with_context_parallel:
        if _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP is None:
            raise RuntimeError("Data parallel group with context parallel combined is not initialized")
        return _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    else:
        if _DATA_PARALLEL_GLOBAL_RANKS is None:
            raise RuntimeError("Data parallel group is not initialized")
        return _DATA_PARALLEL_GLOBAL_RANKS[0]


@subwrold_decorator
def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the current tensor parallel group"""
    if _PIPELINE_GLOBAL_RANKS is None:
        raise RuntimeError("Pipeline parallel group is not initialized")
    return _PIPELINE_GLOBAL_RANKS[0]


@subwrold_decorator
def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the current tensor parallel group"""
    if _PIPELINE_GLOBAL_RANKS is None:
        raise RuntimeError("Pipeline parallel group is not initialized")
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


@subwrold_decorator
def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    if _PIPELINE_GLOBAL_RANKS is None:
        raise RuntimeError("Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


@subwrold_decorator
def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    if _PIPELINE_GLOBAL_RANKS is None:
        raise RuntimeError("Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


@subwrold_decorator
def get_expert_model_parallel_world_size():
    """Return world size for the expert model parallel group"""
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size // get_tensor_model_parallel_world_size()
    else:
        return 0


@subwrold_decorator
def get_expert_model_parallel_rank():
    """Return my rank for the expert parallel group"""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()
    else:
        return 0


@subwrold_decorator
def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    if _GLOBAL_MEMORY_BUFFER is None:
        raise RuntimeError('global memory buffer is not initialized')
    return _GLOBAL_MEMORY_BUFFER


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
