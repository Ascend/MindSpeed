# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

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
    intra_param_end_idx: int # inclusive
    shape: torch.Size
    tensor_model_parallel: bool
    partition_dim : int
    partition_stride : int