# Copyright 2025 Huawei Technologies Co., Ltd.
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
from .replica_group import (
    DUMP_WORLD_GROUP,
    DP_CP_REPLICA_GROUP,
    DP_CP_REPLICA_GROUP_GLOO,
    GLOBAL_DP_CP_RANKS,
    GLOBAL_DP_EP_RANKS,
    REPLICA_NUM,
    set_dump_world_group,
    get_dump_world_group,
    reset_dp_cp_replica_group,
    get_dp_cp_replica_group,
    get_dp_cp_replica_group_gloo,
    set_global_dp_cp_ranks,
    get_global_dp_cp_ranks,
    set_global_dp_ep_ranks,
    get_global_dp_ep_ranks,
    get_replica_num,
    get_dp_cp_ranks,
    get_dp_ranks,
    ttp_get_replica_dp_num,
    ReplicaGroupManager,
    ServerReplicaGroupManager,
)
from .replica_optimizer import TTPReplicaOptimizer
from .replica_optimizer_fp16 import TTPFP16ReplicaOptimizer
from .replica_optimizer_fp32 import TTPFP32ReplicaOptimizer
from .replica_optimizer_chained import TTPReplicaChainedOptimizer

__all__ = [
    'DUMP_WORLD_GROUP',
    'DP_CP_REPLICA_GROUP',
    'DP_CP_REPLICA_GROUP_GLOO',
    'GLOBAL_DP_CP_RANKS',
    'GLOBAL_DP_EP_RANKS',
    'REPLICA_NUM',
    'set_dump_world_group',
    'get_dump_world_group',
    'reset_dp_cp_replica_group',
    'get_dp_cp_replica_group',
    'get_dp_cp_replica_group_gloo',
    'set_global_dp_cp_ranks',
    'get_global_dp_cp_ranks',
    'set_global_dp_ep_ranks',
    'get_global_dp_ep_ranks',
    'get_replica_num',
    'get_dp_cp_ranks',
    'get_dp_ranks',
    'ttp_get_replica_dp_num',
    'ReplicaGroupManager',
    'ServerReplicaGroupManager',
    'TTPReplicaOptimizer',
    'TTPFP16ReplicaOptimizer',
    'TTPFP32ReplicaOptimizer',
    'TTPReplicaChainedOptimizer',
]
