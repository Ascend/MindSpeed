# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from datetime import timedelta
from functools import wraps
from typing import Callable, Optional, List

import torch

_EXPERT_MODEL_PARALLEL_GLOBAL_RANKS = None
_HOT_EXPERT_GROUP_LIST = None


def initialize_model_parallel_impl(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_split_rank: Optional[int] = None,
        pipeline_model_parallel_comm_backend: Optional[str] = None,
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
        create_gloo_process_groups: bool = True,
        create_group: Optional[Callable] = None,
        get_nccl_options: Optional[Callable] = None,
        rank_generator: Optional[Callable] = None,
        get_args: Optional[Callable] = None,
) -> None:
    args = get_args()
    rank = torch.distributed.get_rank()
    # Expert-related parallel groups initialization
    # Build the expert model parallel group
    global _HOT_EXPERT_GROUP_LIST
    if getattr(args, 'trans_hot_expert_group_num', 0) > 0 and _HOT_EXPERT_GROUP_LIST is None:
        _HOT_EXPERT_GROUP_LIST = []
    global _EXPERT_MODEL_PARALLEL_GLOBAL_RANKS
    if encoder_pipeline_model_parallel_size is None:
        encoder_pipeline_model_parallel_size = 0

    if encoder_tensor_model_parallel_size == 0 and encoder_pipeline_model_parallel_size > 0:
        encoder_tensor_model_parallel_size = tensor_model_parallel_size

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

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
        encoder_rank_generator = rank_generator(
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

    decoder_rank_generator = rank_generator(
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

    expert_decoder_rank_generator = rank_generator(
        tp=expert_tensor_parallel_size,
        ep=expert_model_parallel_size,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=1,
        order=order,
        rank_offset=encoder_world_size,
    )

    assert (
            order.endswith("pp")
            or pipeline_model_parallel_size == 1
            or expert_data_parallel_size == data_parallel_size
    ), "When not using pp-last rank ordering, the data parallel size of the attention and moe layers must be the same"

    assert decoder_rank_generator.get_ranks("pp") == expert_decoder_rank_generator.get_ranks(
        "pp"
    ), f"Pipeline parallel groups are expected to be the same for Non-Expert and Expert part, \
    but got {decoder_rank_generator.get_ranks('pp')} and {expert_decoder_rank_generator.get_ranks('pp')}"

    def generator_wrapper(group_type, is_expert=False, **kwargs):
        """The `RankGenerator` class produces a hyper-rectangle for a given set of
        tensor, pipeline, data, expert, and context parallelism. If we have an encoder,
        in addition to the default decoder, we essentially instantiate two `RankGenerator`
        classes to construct the parallelism for each module separately, and we then have
        to stitch them together for the right groups. For now, this means pp and tp-pp.

        Let's say we have a total of 6 GPUs denoted by g0 ... g5.
        For encoder_tp=1, encoder_pp=1, decoder_tp=2, decoder_pp=1, dp=2,
        g0, g1 belong to encoder and g2, ..., g5 belong to decoder.
        The present function will create with "tp-dp-pp":
        3 data-parallel groups: [g0, g1], [g2, g4], [g3, g5]
        4 tensor model-parallel groups: [g0], [g1], [g2, g3], [g4, g5]
        4 pipeline model-parallel groups: [g0, g2], [g0, g3], [g1, g4], [g1, g5]
        """
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
            # Map one encoder tp rank to several decoder tp ranks, because
            # encoder tp and decoder tp won't be the same size.
            # Assign this way to avoid getting the DP ranks mixed up with the PP ranks.
            # For example, if e_ranks = [0,1,2] and d_ranks = [3,4,5,6]
            # Should yield [0,3], [0,4], [1,5], [2,6]
            rep = len(d_ranks) // len(e_ranks)
            remain = len(d_ranks) % len(e_ranks)
            e_ind = 0
            e_rep = rep + int(e_ind < remain)
            for i, y in enumerate(d_ranks):
                x = e_ranks[e_ind]
                e_rep -= 1
                if e_rep == 0:
                    e_ind += 1
                    e_rep = rep + int(e_ind < remain)
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

    for ranks in generator_wrapper('ep', is_expert=True):
        if getattr(args, 'trans_hot_expert_group_num', 0) > 0:
            for _ in range(args.trans_hot_expert_group_num):
                group = create_group(
                    ranks, timeout=timeout, pg_options=get_nccl_options('exp', nccl_comm_cfgs),
                    group_desc='EXPERT_MODEL_PARALLEL_GROUP for broadcast hot expert',
                )
                if rank in ranks:
                    _HOT_EXPERT_GROUP_LIST.append(group)
                    _EXPERT_MODEL_PARALLEL_GLOBAL_RANKS = ranks


def mindspeed_destroy_model_parallel_wrapper(destroy_model_parallel):
    """Wrap the destroy_model_parallel function of megaron."""

    @wraps(destroy_model_parallel)
    def wrapper():
        destroy_model_parallel()
        global _EXPERT_MODEL_PARALLEL_GLOBAL_RANKS
        global _HOT_EXPERT_GROUP_LIST

        _EXPERT_MODEL_PARALLEL_GLOBAL_RANKS = None
        _HOT_EXPERT_GROUP_LIST = None

    return wrapper


def get_expert_model_parallel_global_ranks():
    """Return my rank for the expert model parallel group."""
    global _EXPERT_MODEL_PARALLEL_GLOBAL_RANKS
    return _EXPERT_MODEL_PARALLEL_GLOBAL_RANKS


def get_hot_expert_group_list():
    assert (
            _HOT_EXPERT_GROUP_LIST is not None
    ), 'hot expert group list is not intialized'
    return _HOT_EXPERT_GROUP_LIST
