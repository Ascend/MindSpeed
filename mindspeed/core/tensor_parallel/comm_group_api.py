# Copyright 2024 Huawei Technologies Co., Ltd
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
# ============================================================================
import abc

from megatron.core.parallel_state import get_context_parallel_group
from megatron.core.parallel_state import get_context_parallel_rank
from megatron.core.parallel_state import get_context_parallel_world_size
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron.core.parallel_state import get_tensor_model_parallel_world_size

from mindspeed.core.parallel_state import get_tensor_model_parallel_group_for_nd1_dim1
from mindspeed.core.parallel_state import get_tensor_model_parallel_group_for_nd1_dim1_rank
from mindspeed.core.parallel_state import get_tensor_model_parallel_group_for_nd1_dim1_world_size
from mindspeed.core.parallel_state import get_tensor_model_parallel_group_for_nd1_dim2
from mindspeed.core.parallel_state import get_tensor_model_parallel_group_for_nd1_dim2_rank
from mindspeed.core.parallel_state import get_tensor_model_parallel_group_for_nd1_dim2_world_size
from mindspeed.core.parallel_state import get_tp_x_ep_group
from mindspeed.core.parallel_state import get_tp_x_ep_group_rank
from mindspeed.core.parallel_state import get_tp_x_ep_group_world_size
from mindspeed.core.parallel_state import get_tp_x_ring_global_ranks
from mindspeed.core.parallel_state import get_tp_x_sd_rcv_overlap_group
from mindspeed.core.parallel_state import get_tp_y_ring_global_ranks
from mindspeed.core.parallel_state import get_tp_y_sd_rcv_overlap_group


class CollectiveCommIntf:
    def __init__(self, comm_group_name):
        self.comm_group_name = comm_group_name

    @classmethod
    @abc.abstractmethod
    def get_comm_group_world_size(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_comm_group(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_comm_rank(cls):
        raise NotImplementedError

    def get_comm_group_name(self):
        return self.comm_group_name


class OverlapCollectiveIntf(CollectiveCommIntf):
    @classmethod
    @abc.abstractmethod
    def get_ring_global_ranks(cls):
        raise NotImplementedError


class CPCollectiveComm(CollectiveCommIntf):
    @classmethod
    def get_comm_group_world_size(cls):
        return get_context_parallel_world_size()

    @classmethod
    def get_comm_group(cls):
        return get_context_parallel_group()

    @classmethod
    def get_comm_rank(cls):
        return get_context_parallel_rank()


class TPXCollectiveComm(CollectiveCommIntf):
    def __init__(self, name="tp-x"):
        super().__init__(name)

    @classmethod
    def get_comm_rank(cls):
        return get_tensor_model_parallel_group_for_nd1_dim1_rank()

    @classmethod
    def get_comm_group_world_size(cls):
        return get_tensor_model_parallel_group_for_nd1_dim1_world_size()

    @classmethod
    def get_comm_group(cls):
        return get_tensor_model_parallel_group_for_nd1_dim1()


class TPXEPCollectiveComm(CollectiveCommIntf):
    def __init__(self, name="tp-x-ep"):
        super().__init__(name)

    @classmethod
    def get_comm_rank(cls):
        return get_tp_x_ep_group_rank()

    @classmethod
    def get_comm_group_world_size(cls):
        return get_tp_x_ep_group_world_size()

    @classmethod
    def get_comm_group(cls):
        return get_tp_x_ep_group()


class TPXOverlapCollectiveComm(TPXCollectiveComm, OverlapCollectiveIntf):
    def __init__(self):
        super().__init__("tp-x-overlap")

    @classmethod
    def get_comm_group(cls):
        return get_tp_x_sd_rcv_overlap_group()

    @classmethod
    def get_ring_global_ranks(cls):
        return get_tp_x_ring_global_ranks()


class TPYCollectiveComm(CollectiveCommIntf):
    def __init__(self, name="tp-y"):
        super().__init__(name)

    @classmethod
    def get_comm_rank(cls):
        return get_tensor_model_parallel_group_for_nd1_dim2_rank()

    @classmethod
    def get_comm_group_world_size(cls):
        return get_tensor_model_parallel_group_for_nd1_dim2_world_size()

    @classmethod
    def get_comm_group(cls):
        return get_tensor_model_parallel_group_for_nd1_dim2()


class TPYOverlapCollectiveComm(TPYCollectiveComm, OverlapCollectiveIntf):
    def __init__(self):
        super().__init__("y-overlap")

    @classmethod
    def get_comm_group(cls):
        return get_tp_y_sd_rcv_overlap_group()

    @classmethod
    def get_ring_global_ranks(cls):
        return get_tp_y_ring_global_ranks()


class TPXYCollectiveComm(CollectiveCommIntf):
    def __init__(self):
        super().__init__("tp-xy")

    @classmethod
    def get_comm_rank(cls):
        return get_tensor_model_parallel_rank()

    @classmethod
    def get_comm_group_world_size(cls):
        return get_tensor_model_parallel_world_size()

    @classmethod
    def get_comm_group(cls):
        return get_tensor_model_parallel_group()
