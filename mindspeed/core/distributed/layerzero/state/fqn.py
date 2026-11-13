# Copyright 2024 Huawei Technologies Co., Ltd.
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

from dataclasses import dataclass
import torch

ITERATION_KEY = "iteration"
ARGS_KEY = "args"
LOCAL_NAME_TO_FQN_KEY = "shard_state_dict"
D3PARALLEL_KEY = ""
MODEL_SD_KEY = "model"


@dataclass
class ShardFlattenInfo:
    '''
    This class is unsed for saving flatten shard parameter global info
    and helps to convert full param into shard param

    [offset, offset+numel]
    '''
    in_shard: bool
    numel: int
    intra_param_start_idx: int
    intra_param_end_idx: int  # inclusive
    shape: torch.Size
    tensor_model_parallel: bool
    partition_dim: int
    partition_stride: int
