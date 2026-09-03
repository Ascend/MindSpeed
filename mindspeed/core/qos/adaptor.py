# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import wraps
from logging import getLogger

import torch_npu

from mindspeed.args_utils import get_full_args as get_args
from mindspeed.core.qos.qos import Qos
from mindspeed.log_config import log_rank_0
from mindspeed.ops.npu_matmul_add import get_npu_version, NPUVersion

LOG = getLogger(__name__)

# Map get_nccl_options(pg_name, ...) to QoS parallel_type.
# Covers Megatron create_group path and MindSpeed direct new_group paths.
PG_NAME_TO_PARALLEL_TYPE = {
    # Megatron core groups
    'dp': 'dp',
    'dp_cp': 'dp-cp',
    'intra_dp_cp': 'dp-cp',
    'inter_dp_cp': 'dp-cp',
    'cp': 'cp',
    'hcp': 'cp',
    'mp': 'mp',
    'tp': 'tp',
    'pp': 'pp',
    'embd': 'embd',
    'pos_embd': 'pos-embd',
    'tp_dp_cp': 'tp-dp-cp',
    'tp_dp': 'tp-dp',
    'tp_cp': 'tp-cp',
    'ep': 'ep',
    'ep_tp': 'ep-tp',
    'tp_ep_mp': 'tp-ep-mp',
    'tp_ep_pp': 'tp-ep-pp',
    'ep_dp': 'ep-dp',
    # Megatron 0.18 hierarchical EP-DP and distributed-optimizer groups
    'intra_ep_dp': 'ep-dp',
    'inter_ep_dp': 'ep-dp',
    'intra_dist_opt_instance': 'dp',
    # MindSpeed extensions that call torch.distributed.new_group directly
    'cp2': 'cp',
    'cp_ulysses': 'cp',
    'cp_ring': 'cp',
    'cp_ring_intra': 'cp',
    'cp_ring_intra_overlap': 'cp',
    'pp_new_stream': 'pp',
    'exp': 'ep',
    'tp_exp': 'tp-ep-mp',
    'inner_dp': 'dp',
    'nd1_dim1': 'tp',
    'nd1_dim2': 'tp',
    'nd2_dim1': 'tp',
    'nd2_dim2': 'tp',
    'ag_x_sd_rcv_overlap': 'tp',
    'ag_y_sd_rcv_overlap': 'tp',
}


def resolve_parallel_type(pg_name):
    if pg_name is None:
        return None
    return PG_NAME_TO_PARALLEL_TYPE.get(pg_name)


def apply_qos_to_pg_options(pg_options, parallel_type):
    """Inject QoS fields into ProcessGroupHCCL options. Returns the options object."""
    ai_qos = Qos()
    roce_qos = ai_qos.set_parallel_roce_qos(parallel_type)
    hccl_qos = ai_qos.set_parallel_hccl_qos(parallel_type)
    if not (0 <= roce_qos <= 7) or not (0 <= hccl_qos <= 7):
        error_msg_parts = []
        if not (0 <= roce_qos <= 7):
            error_msg_parts.append(f"roce_qos={roce_qos} (valid range: 0-7)")
        if not (0 <= hccl_qos <= 7):
            error_msg_parts.append(f"hccl_qos={hccl_qos} (valid range: 0-7)")
        raise ValueError(f"Invalid QoS value for parallel type '{parallel_type}'! " + " | ".join(error_msg_parts))

    qos_config = {}
    args = get_args()
    if get_npu_version() in (NPUVersion.A3, NPUVersion.A5):
        qos_config['hccl_sdma_qos'] = hccl_qos
        if args.aiqos_enable_roce:
            qos_config['qos_service_level'] = roce_qos
            qos_config['qos_traffic_class'] = roce_qos * 32
            log_rank_0(LOG.info, f"{parallel_type} roce_qos: {roce_qos}, hccl_qos: {hccl_qos}")
        else:
            log_rank_0(LOG.info, f"{parallel_type} hccl_qos: {hccl_qos}")
    else:
        if args.aiqos_enable_roce:
            qos_config['qos_service_level'] = roce_qos
            qos_config['qos_traffic_class'] = roce_qos * 32
            log_rank_0(LOG.info, f"{parallel_type} roce_qos: {roce_qos}")

    if not qos_config:
        return pg_options
    if pg_options is None or not hasattr(pg_options, 'hccl_config'):
        pg_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()

    # Preserve settings such as hccl_buffer_size that other features may have
    # already written into the same process-group options object.
    merged_config = dict(getattr(pg_options, 'hccl_config', None) or {})
    merged_config.update(qos_config)
    pg_options.hccl_config = merged_config
    return pg_options


def get_nccl_options_qos_wrapper(get_nccl_options):
    """Wrap Megatron's process-group option builder to inject HCCL QoS."""

    @wraps(get_nccl_options)
    def wrapper(pg_name, nccl_comm_cfgs):
        options = get_nccl_options(pg_name, nccl_comm_cfgs)

        parallel_type = resolve_parallel_type(pg_name)
        if parallel_type is None:
            return options

        return apply_qos_to_pg_options(options, parallel_type)

    return wrapper
