# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from logging import getLogger
from typing import (
    Optional,
    Callable,
    List
)
from functools import partial
from datetime import timedelta

import torch
import torch_npu

from megatron.core.utils import GlobalMemoryBuffer, is_torch_min_version
from megatron.core.parallel_state import (
    default_embedding_ranks,
    default_position_embedding_ranks,
    RankGenerator,
    create_group,
    get_nccl_options,
    get_data_parallel_group,
    create_hierarchical_parallel_groups,
    _set_global_memory_buffer
)

import megatron.core.parallel_state as mcps
from mindspeed.core.qos.qos import Qos
from mindspeed.log_config import log_rank_0

LOG = getLogger(__name__)


def create_group_qos(
        ranks=None,
        timeout=None,
        backend=None,
        pg_options=None,
        use_local_synchronization=False,
        group_desc=None,
        parallel_type=None
):
    """Creates a Qos ProcessGroup."""
    kwargs = {
        'ranks': ranks,
        'timeout': timeout,
        'backend': backend,
        'pg_options': pg_options,
        'use_local_synchronization': use_local_synchronization,
        'group_desc': group_desc,
    }
    if not is_torch_min_version('2.4.0'):
        kwargs.pop('group_desc')
        if timeout is None:
            # Old version (e.g. v2.1.2) sets default_pg_timeout as default value to timeout
            # in function signature, then check tiemout value type.
            # New version sets None as default value to timeout in function signature. If value
            # is None, torch will give value according to the backend, then check type.
            # So need to unset timeout here if caller doesn't set value. Otherwise there is
            # type error.
            kwargs.pop('timeout')
    if pg_options is None:
        kwargs['pg_options'] = torch_npu._C._distributer_c10d.ProcessGroupHCCL.Options()
    ai_qos = Qos()
    qos = ai_qos.set_parallel_qos(parallel_type)
    if not (0 <= qos <= 7):
        raise ValueError(f"Invalid QoS value! {parallel_type}={qos} (valid range: 0-7)")
    kwargs['pg_options'].hccl_config = {'qos_service_level': qos, 'qos_traffic_class': qos * 32}
    log_rank_0(LOG.info, f"{parallel_type} ai qos: {qos}")
    return torch.distributed.new_group(**kwargs)


def create_hierarchical_parallel_groups_qos(
        rank, ranks, group_size, hierarchical_group_sizes, pg_options, parallel_type
):
    """Create hierarchical groups for one parallelism.
    Taking a group size of 16 as example, so we have a total of 16 GPUs denoted by g0 ... g15.
    If the hierarchical group sizes are [2,2,4], we use 2 GPUs in the first and second level
    of sub-groups, and 4 GPUs in the last level of sub groups. The present function will
    create 8 level-1 sub-groups, 8 level-2 sub-groups and 4 level-3 sub-groups as:
        8 level-1 sub-groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        8 level-2 sub-groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        4 level-3 sub-groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    """

    hierarchical_groups = []
    accumulated_group_sizes = 1
    processed_group_sizes = 1
    for level, hierarchical_group_size in enumerate(hierarchical_group_sizes):
        accumulated_group_sizes *= hierarchical_group_size
        for k in range(group_size // accumulated_group_sizes):
            for j in range(processed_group_sizes):
                global_sub_ranks = [
                    ranks[j + i * processed_group_sizes + k * accumulated_group_sizes]
                    for i in range(hierarchical_group_size)
                ]
                sub_group = create_group_qos(
                    global_sub_ranks,
                    pg_options=pg_options,
                    group_desc=f'HIERARCHICAL_CONTEXT_PARALLEL_GROUP_L{level}',
                    parallel_type=parallel_type
                )
                if rank in global_sub_ranks:
                    hierarchical_groups.append(sub_group)
        processed_group_sizes *= hierarchical_group_size
    return hierarchical_groups


def initialize_model_parallel_qos(
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
) -> None:
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
        mcps._PIPELINE_MODEL_PARALLEL_DECODER_START = encoder_pipeline_model_parallel_size

    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed has not been initialized yet.")
    world_size: int = torch.distributed.get_world_size()

    if encoder_tensor_model_parallel_size > 0:
        if not (encoder_tensor_model_parallel_size <= tensor_model_parallel_size):
            raise RuntimeError(
                "encoder_tensor_model_parallel_size must be less than or equal to tensor_model_parallel_size.")

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

    if not (encoder_world_size + decoder_world_size == world_size):
        raise RuntimeError(f"{encoder_world_size=} + {decoder_world_size=} != {world_size=}")

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 1:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 1 with interleaved schedule"
            )
        mcps._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        mcps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        mcps._PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

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

    expert_decoder_rank_generator = RankGenerator(
        tp=expert_tensor_parallel_size,
        ep=expert_model_parallel_size,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=1,
        order=order,
        rank_offset=encoder_world_size,
    )

    condition = (
            order.endswith("pp")
            or pipeline_model_parallel_size == 1
            or expert_data_parallel_size == data_parallel_size
    )
    if not condition:
        raise RuntimeError(
            "When not using pp-last rank ordering, the data parallel size of the attention and moe layers must be the same"
        )

    decoder_pp_ranks = decoder_rank_generator.get_ranks("pp")
    expert_decoder_pp_ranks = expert_decoder_rank_generator.get_ranks("pp")

    if not (decoder_pp_ranks == expert_decoder_pp_ranks):
        raise RuntimeError(
            f"Pipeline parallel groups are expected to be the same for Non-Expert and Expert part, "
            f"but got {decoder_pp_ranks} and {expert_decoder_pp_ranks}"
        )

    def generator_wrapper(group_type, is_expert=False, **kwargs):
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
            rep = len(d_ranks) // len(e_ranks)
            remain = len(d_ranks) % len(e_ranks)
            e_ind = 0
            e_rep = rep + int(e_ind < remain)
            for _, y in enumerate(d_ranks):
                x = e_ranks[e_ind]
                e_rep -= 1
                if e_rep == 0:
                    e_ind += 1
                    e_rep = rep + int(e_ind < remain)
                yield x + y
        elif group_type == 'tp-pp':
            if not (len(e_ranks) == len(d_ranks)):
                raise RuntimeError(
                    f"The length of e_ranks ({len(e_ranks)}) does not match the length of d_ranks ({len(d_ranks)}).")
            for x, y in zip(e_ranks, d_ranks):
                yield x + y
        else:
            for x in e_ranks:
                yield x
            for x in d_ranks:
                yield x

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    if not (mcps._DATA_PARALLEL_GROUP is None):
        raise RuntimeError('data parallel group is already initialized')

    for ranks in generator_wrapper('dp'):
        group = create_group_qos(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('dp', nccl_comm_cfgs),
            group_desc='DATA_PARALLEL_GROUP',
            parallel_type='dp'
        )
        if create_gloo_process_groups:
            group_gloo = create_group(
                ranks, timeout=timeout, backend="gloo", group_desc='DATA_PARALLEL_GROUP_GLOO'
            )
        else:
            group_gloo = None
        if rank in ranks:
            mcps._DATA_PARALLEL_GROUP = group
            mcps._DATA_PARALLEL_GROUP_GLOO = group_gloo
            mcps._DATA_PARALLEL_GLOBAL_RANKS = ranks

    if not ((data_parallel_size * context_parallel_size) % num_distributed_optimizer_instances == 0):
        raise RuntimeError(
            'Data parallel size should be divisible by partial DistOpt shard factor'
        )
    intra_partial_data_parallel_size = (
                                               data_parallel_size * context_parallel_size
                                       ) // num_distributed_optimizer_instances

    for ranks_with_cp in generator_wrapper('dp-cp'):
        group_with_cp = create_group_qos(
            ranks_with_cp,
            timeout=timeout,
            pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs),
            group_desc='DATA_PARALLEL_GROUP_WITH_CP',
            parallel_type='dp-cp'
        )
        if create_gloo_process_groups:
            group_with_cp_gloo = create_group(
                ranks_with_cp,
                timeout=timeout,
                backend="gloo",
                group_desc='DATA_PARALLEL_GROUP_WITH_CP_GLOO',
            )
        else:
            group_with_cp_gloo = None
        if rank in ranks_with_cp:
            mcps._DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
            mcps._DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
            mcps._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

        if num_distributed_optimizer_instances > 1:
            for i in range(num_distributed_optimizer_instances):
                intra_partial_data_parallel_ranks_with_cp = ranks_with_cp[
                    (i * intra_partial_data_parallel_size): (
                            (i + 1) * intra_partial_data_parallel_size
                    )
                ]

                intra_partial_data_parallel_group_with_cp = create_group_qos(
                    intra_partial_data_parallel_ranks_with_cp,
                    timeout=timeout,
                    pg_options=get_nccl_options('intra_dp_cp', nccl_comm_cfgs),
                    group_desc='INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP',
                    parallel_type='intra-dp-cp'
                )
                if create_gloo_process_groups:
                    intra_partial_data_parallel_group_with_cp_gloo = create_group(
                        intra_partial_data_parallel_ranks_with_cp,
                        timeout=timeout,
                        backend="gloo",
                        group_desc='INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO',
                    )
                else:
                    intra_partial_data_parallel_group_with_cp_gloo = None

                if rank in intra_partial_data_parallel_ranks_with_cp:
                    mcps._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = (
                        intra_partial_data_parallel_group_with_cp
                    )
                    mcps._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = (
                        intra_partial_data_parallel_group_with_cp_gloo
                    )

            for i in range(intra_partial_data_parallel_size):
                inter_partial_data_parallel_ranks_with_cp = ranks_with_cp[
                    i::intra_partial_data_parallel_size
                ]

                inter_partial_data_parallel_group_with_cp = create_group_qos(
                    inter_partial_data_parallel_ranks_with_cp,
                    timeout=timeout,
                    pg_options=get_nccl_options('inter_dp_cp', nccl_comm_cfgs),
                    group_desc='INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP',
                    parallel_type='inter-dp-cp'
                )

                if rank in inter_partial_data_parallel_ranks_with_cp:
                    mcps._INTER_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = (
                        inter_partial_data_parallel_group_with_cp
                    )
        else:
            mcps._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = mcps._DATA_PARALLEL_GROUP_WITH_CP
            mcps._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = mcps._DATA_PARALLEL_GROUP_WITH_CP_GLOO

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
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # Build the context-parallel groups.
    if not (mcps._CONTEXT_PARALLEL_GROUP is None):
        raise RuntimeError('context parallel group is already initialized')
    for ranks in generator_wrapper('cp'):
        group = create_group_qos(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('cp', nccl_comm_cfgs),
            group_desc='CONTEXT_PARALLEL_GROUP',
            parallel_type='cp'
        )
        if rank in ranks:
            mcps._CONTEXT_PARALLEL_GROUP = group
            mcps._CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
        if hierarchical_context_parallel_sizes:
            mcps._HIERARCHICAL_CONTEXT_PARALLEL_GROUPS += create_hierarchical_parallel_groups_qos(
                rank,
                ranks,
                context_parallel_size,
                hierarchical_context_parallel_sizes,
                get_nccl_options('hcp', nccl_comm_cfgs),
                parallel_type='hcp'
            )

    # Build the model-parallel groups.
    if not (mcps._MODEL_PARALLEL_GROUP is None):
        raise RuntimeError('model parallel group is already initialized')
    for ranks in generator_wrapper('tp-pp'):
        group = create_group_qos(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('mp', nccl_comm_cfgs),
            group_desc='MODEL_PARALLEL_GROUP',
            parallel_type='mp'
        )
        if rank in ranks:
            mcps._MODEL_PARALLEL_GROUP = group
            mcps._MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Build the tensor model-parallel groups.
    if not (mcps._TENSOR_MODEL_PARALLEL_GROUP is None):
        raise RuntimeError('tensor model parallel group is already initialized')
    for ranks in generator_wrapper('tp'):
        group = create_group_qos(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('tp', nccl_comm_cfgs),
            group_desc='TENSOR_MODEL_PARALLEL_GROUP',
            parallel_type='tp'
        )
        if rank in ranks:
            mcps._TENSOR_MODEL_PARALLEL_GROUP = group
            mcps._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Build the pipeline model-parallel groups and embedding groups
    if not (mcps._PIPELINE_MODEL_PARALLEL_GROUP is None):
        raise RuntimeError('pipeline model parallel group is already initialized')
    if not (mcps._EMBEDDING_GROUP is None):
        raise RuntimeError('embedding group is already initialized')
    if not (mcps._POSITION_EMBEDDING_GROUP is None):
        raise RuntimeError('position embedding group is already initialized')

    if pipeline_model_parallel_comm_backend == 'ucc':
        if 'CUDA_DEVICE_MAX_CONNECTIONS' in os.environ:
            if not (os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] != '1'):
                raise RuntimeError("UCC-backend requires CUDA_DEVICE_MAX_CONNECTIONS > 1")

        os.environ['TORCH_UCC_BLOCKING_WAIT'] = (
            os.environ['TORCH_UCC_BLOCKING_WAIT']
            if "TORCH_UCC_BLOCKING_WAIT" in os.environ
            else 'none'
        )
        os.environ['UCC_EC_CUDA_STREAM_TASK_MODE'] = (
            os.environ['UCC_EC_CUDA_STREAM_TASK_MODE']
            if "UCC_EC_CUDA_STREAM_TASK_MODE" in os.environ
            else 'driver'
        )
        os.environ['UCX_TLS'] = (
            os.environ['UCX_TLS'] if "UCX_TLS" in os.environ else 'ib,cuda_copy'
        )
        os.environ['NSYS_UCP_COMM_PARAMS'] = '1'
        os.environ['UCX_RNDV_THRESH'] = '0'
        os.environ['UCX_NET_DEVICES'] = 'all'
        os.environ['UCC_CL_BASIC_TLS'] = '^sharp,nccl'

    for ranks in generator_wrapper('pp'):
        group = create_group_qos(
            ranks,
            timeout=timeout,
            backend=pipeline_model_parallel_comm_backend,
            pg_options=(
                None
                if pipeline_model_parallel_comm_backend == 'ucc'
                else get_nccl_options('pp', nccl_comm_cfgs)
            ),
            group_desc='PIPELINE_MODEL_PARALLEL_GROUP',
            parallel_type='pp'
        )
        if not (
                pipeline_model_parallel_comm_backend is None
                or pipeline_model_parallel_comm_backend == 'nccl'
                or pipeline_model_parallel_comm_backend == 'ucc'
        ):
            raise RuntimeError(
                f'"{pipeline_model_parallel_comm_backend}" backend for PP communication is currently not supported')

        if rank in ranks:
            if mcps._PIPELINE_MODEL_PARALLEL_GROUP is None:
                mcps._PIPELINE_MODEL_PARALLEL_GROUP = group
                mcps._PIPELINE_GLOBAL_RANKS = ranks
            elif isinstance(mcps._PIPELINE_GLOBAL_RANKS[0], list):
                mcps._PIPELINE_MODEL_PARALLEL_GROUP.append(group)
                mcps._PIPELINE_GLOBAL_RANKS.append(ranks)
            else:
                mcps._PIPELINE_MODEL_PARALLEL_GROUP = [mcps._PIPELINE_MODEL_PARALLEL_GROUP, group]
                mcps._PIPELINE_GLOBAL_RANKS = [mcps._PIPELINE_GLOBAL_RANKS, ranks]

        embedding_ranks = get_embedding_ranks(ranks)
        group = create_group_qos(
            embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options('embd', nccl_comm_cfgs),
            group_desc='EMBEDDING_GROUP',
            parallel_type='embd'
        )
        if rank in embedding_ranks:
            mcps._EMBEDDING_GROUP = group
            mcps._EMBEDDING_GLOBAL_RANKS = embedding_ranks

        position_embedding_ranks = get_position_embedding_ranks(ranks)
        group = create_group_qos(
            position_embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options('pos_embd', nccl_comm_cfgs),
            group_desc='POSITION_EMBEDDING_GROUP',
            parallel_type='pos-embd'
        )
        if rank in position_embedding_ranks:
            mcps._POSITION_EMBEDDING_GROUP = group
            mcps._POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the tensor + data parallel groups.
    if not (mcps._TENSOR_AND_DATA_PARALLEL_GROUP is None):
        raise RuntimeError('Tensor + data parallel group is already initialized')
    for ranks in generator_wrapper('tp-dp-cp'):
        group = create_group_qos(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs),
            group_desc='TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP',
        )
        if rank in ranks:
            mcps._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group
    for ranks in generator_wrapper('tp-dp'):
        group = create_group_qos(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs),
            group_desc='TENSOR_AND_DATA_PARALLEL_GROUP',
            parallel_type='tp-dp'
        )
        if rank in ranks:
            mcps._TENSOR_AND_DATA_PARALLEL_GROUP = group

    # Build the tensor + context parallel groups.
    if not (mcps._TENSOR_AND_CONTEXT_PARALLEL_GROUP is None):
        raise RuntimeError('Tensor + context parallel group is already initialized')
    for ranks in generator_wrapper('tp-cp'):
        group = create_group_qos(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('tp_cp', nccl_comm_cfgs),
            group_desc='TENSOR_AND_CONTEXT_PARALLEL_GROUP',
            parallel_type='tp-cp'
        )
        if rank in ranks:
            mcps._TENSOR_AND_CONTEXT_PARALLEL_GROUP = group

    ### Expert-related parallel groups initialization
    # Build the expert model parallel group
    if not (mcps._EXPERT_MODEL_PARALLEL_GROUP is None):
        raise RuntimeError('Expert parallel group is already initialized')
    for ranks in generator_wrapper('ep', is_expert=True):
        group = create_group_qos(
            ranks,
            pg_options=get_nccl_options('ep', nccl_comm_cfgs),
            group_desc='EXPERT_MODEL_PARALLEL_GROUP',
            parallel_type='ep'
        )
        if rank in ranks:
            mcps._EXPERT_MODEL_PARALLEL_GROUP = group

    # Build the expert tensor parallel group
    if not (mcps._EXPERT_TENSOR_PARALLEL_GROUP is None):
        raise RuntimeError('Expert tensor model parallel group is already initialized')
    for ranks in generator_wrapper('tp', is_expert=True):
        group = create_group_qos(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('ep_tp', nccl_comm_cfgs),
            group_desc='EXPERT_TENSOR_PARALLEL_GROUP',
            parallel_type='ep-tp'
        )
        if rank in ranks:
            mcps._EXPERT_TENSOR_PARALLEL_GROUP = group

    # Build the tensor + expert parallel groups
    if not (mcps._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is None):
        raise RuntimeError('Expert tensor + model parallel group is already initialized')
    for ranks in generator_wrapper('tp-ep', is_expert=True):
        group = create_group_qos(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('tp_ep_mp', nccl_comm_cfgs),
            group_desc='EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP',
            parallel_type='tp-ep-mp'
        )
        if rank in ranks:
            mcps._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = group

    # Build the expert+tensor+pipeline parallel groups
    if not (mcps._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is None):
        raise RuntimeError('The expert_tensor_model_pipeline parallel group is already initialized')
    for ranks in generator_wrapper('tp-ep-pp', is_expert=True):
        group = create_group_qos(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('tp_ep_pp', nccl_comm_cfgs),
            group_desc='EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP',
            parallel_type='tp-ep-pp'
        )
        if rank in ranks:
            mcps._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = group

    # Build the expert data parallel group
    if not (mcps._EXPERT_DATA_PARALLEL_GROUP is None):
        raise RuntimeError('Expert data group is already initialized')
    if not (mcps._EXPERT_DATA_PARALLEL_GROUP_GLOO is None):
        raise RuntimeError('Expert data group-gloo is already initialized')

    for ranks in generator_wrapper('dp', is_expert=True):
        group = create_group_qos(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('ep_dp', nccl_comm_cfgs),
            group_desc='EXPERT_DATA_PARALLEL_GROUP',
            parallel_type='ep-dp'
        )
        if create_gloo_process_groups:
            group_gloo = create_group(
                ranks, backend="gloo", group_desc='EXPERT_DATA_PARALLEL_GROUP_GLOO'
            )
        else:
            group_gloo = None
        if rank in ranks:
            mcps._EXPERT_DATA_PARALLEL_GROUP = group
            mcps._EXPERT_DATA_PARALLEL_GROUP_GLOO = group_gloo
    ### End of expert related parallel groups initialization

    # Initialize global memory buffer
    _set_global_memory_buffer()
