from typing import Optional, List, Dict
import torch
import torch_npu
from torch.library import impl
from mindspeed.op_builder import AllToAllAllGatherBatchMatMulOpBuilder
from mindspeed.op_builder.builder import AS_LIBRARY

__all__ = ["npu_alltoall_allgather_bmm"]


mindspeed_ops_builder = AllToAllAllGatherBatchMatMulOpBuilder()
SUPPORTED_ACT_TYPE = ["None", "GeLu", "SiLu", "ReLu", "FastGeLu"]
ACT_TYPE_DICT = {"none": 0,
                "gelu": 1,
                "silu": 2,
                "relu": 3,
                "fastgelu": 4
                }


@impl(AS_LIBRARY, "npu_alltoall_allgather_bmm", "PrivateUse1")
def npu_alltoall_allgather_bmm_single(x,
                                  weight,
                                  group_ep,
                                  group_ep_worldsize,
                                  group_tp,
                                  group_tp_worldsize,
                                  *,
                                  bias=None,
                                  shard_type=0,
                                  act_type=0,
                                  need_allgather_out=False,
                                  need_activation_feature=False):
    mindspeed_ops = mindspeed_ops_builder.load()
    outputs = mindspeed_ops.npu_alltoall_allgather_bmm(x, weight, bias,
                                                       group_ep, group_ep_worldsize,
                                                       group_tp, group_tp_worldsize,
                                                       shard_type, act_type,
                                                       need_allgather_out,
                                                       need_activation_feature)
    return outputs


def convert_act_type(act_type):
    if not isinstance(act_type, str):
        raise AssertionError(f'act_type should be str type, but got type {type(act_type)}')
    act_type_lower = act_type.lower()
    if act_type_lower in ACT_TYPE_DICT:
        return ACT_TYPE_DICT[act_type_lower]
    raise AssertionError(f'Unknown act_type: {act_type}, supported act_type:{SUPPORTED_ACT_TYPE}, case insensitive')


def npu_alltoall_allgather_bmm(*args, **kwargs):
    if 'act_type' not in kwargs:
        kwargs['act_type'] = 0
    else:
        kwargs['act_type'] = convert_act_type(kwargs['act_type'])
    return torch.ops.mindspeed.npu_alltoall_allgather_bmm(*args, **kwargs)