# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""MindSpeed TTP (Training Tolerance Platform)"""

from .config import TTPConfig, HeartbeatConfig, get_ttp_config, set_ttp_config
from .constants import WorkerStatus, MsgType

from .comm.controller import TTPController
from .comm.processor import TTPProcessor, register_save_ckpt_handler, get_processor, get_worker_instance

from .recovery.dump_save import (
    tft_init_controller_processor,
    tft_register_processor,
    tft_save_callback,
    tft_set_optimizer_replica,
)
from .recovery.exception_handler import (
    ttp_exception_handler,
    _send_stopped_heartbeat_and_exit,
)
from .utils.worker_utils import (
    _get_actor_module_from_worker,
    _get_actor_optimizer_from_worker,
    _is_actor_param_offload,
    _get_checkpoint_manager_from_worker,
)
from .replica.replica_group import (
    get_dp_cp_ranks,
    get_dp_ranks,
    get_global_dp_cp_ranks,
    get_global_dp_ep_ranks,
    set_global_dp_cp_ranks,
    set_global_dp_ep_ranks,
    get_replica_num,
    ttp_get_replica_dp_num,
    get_dp_cp_replica_group_gloo,
)

__all__ = [
    'TTPConfig',
    'HeartbeatConfig',
    'get_ttp_config',
    'set_ttp_config',
    'WorkerStatus',
    'MsgType',
    'TTPController',
    'TTPProcessor',
    'register_save_ckpt_handler',
    'get_processor',
    'get_worker_instance',
    'tft_init_controller_processor',
    'tft_register_processor',
    'tft_save_callback',
    'tft_set_optimizer_replica',
    'ttp_exception_handler',
    'get_dp_cp_ranks',
    'get_dp_ranks',
    'get_global_dp_cp_ranks',
    'get_global_dp_ep_ranks',
    'set_global_dp_cp_ranks',
    'set_global_dp_ep_ranks',
    'get_replica_num',
    'ttp_get_replica_dp_num',
    'get_dp_cp_replica_group_gloo',
    'ttp_exception_handler',
    '_send_stopped_heartbeat_and_exit',
    '_get_actor_module_from_worker',
    '_get_actor_optimizer_from_worker',
    '_is_actor_param_offload',
    '_get_checkpoint_manager_from_worker',
]
