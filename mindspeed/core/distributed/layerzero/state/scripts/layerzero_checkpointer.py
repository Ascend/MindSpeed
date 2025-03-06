# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

import os
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import OrderedDict, defaultdict

import torch
from mindspeed.core.distributed.layerzero.zero3._common_utils import (
    clean_tensor_name,
)

ITERATION_KEY = "iteration"
ARGS_KEY = "args"
LOCAL_NAME_TO_FQN_KEY = "shard_state_dict"
PARALLE_STATE_KAY = "parallel_state"
MODEL_SD_KEY = "model"
PP_LAYER_PATTERN = re.compile(r"(layers\.)(\d+)(\..*)")

MODEL_FILE_KEY = "model_"
NUM_LAYERS_KEY = "num_layers"
PP_LAYERS_KEY = "layers_per_pp"

EMA_MODEL_SD_KEY = "ema_model"
MODEL_PREFIX = None


def remove_model_prefix(prefix):
    print(f"[debug] set model prefix =", prefix)
    global MODEL_PREFIX
    if prefix:
        MODEL_PREFIX = prefix + '.'
    

def clean_prefix(fqn, prefix):
    if prefix:
        fqn = fqn.replace(prefix, "")
    return fqn


def set_ema_model():
    global MODEL_SD_KEY
    global EMA_MODEL_SD_KEY
    MODEL_SD_KEY = EMA_MODEL_SD_KEY


class ShardStateDict:

    def __init__(self, filename) -> None:
        self.filename = filename
        self._init_metadata()

    def _init_metadata(self):
        state_dict = torch.load(self.filename, map_location='cpu')

        self.parallel_info = state_dict[PARALLE_STATE_KAY]
        self._param_key_to_shard_info = state_dict[LOCAL_NAME_TO_FQN_KEY]
        self.model_state_dict = state_dict[MODEL_SD_KEY]

        self.tp_rank = self.parallel_info["tp_rank"]
        self.pp_rank = self.parallel_info["pp_rank"]
        self.global_rank = self.parallel_info["global_rank"]
        self.tp_degree = self.parallel_info["tp_degree"]
        self.pp_degree = self.parallel_info["pp_degree"]
        self.dp_degree = self.parallel_info["dp_degree"]

    def _get_param_by_param_key(self, param_key) -> torch.Tensor:
        param = self.model_state_dict.get(param_key, None)
        return param

    def _get_shape_by_param_key(self, key: str) -> torch.Tensor:
        shard_info = self._get_shard_info_by_fqn(key)
        return shard_info.shape

    def _get_tp_pp_rank(self) -> Tuple[int, int]:
        return (self.tp_rank, self.pp_rank)

    def __lt__(self, rhs):
        return self.global_rank < rhs.global_rank

    def __len__(self):
        return len(self.model_state_dict)

    def _get_shard_info_by_fqn(self, key: str):
        shard_info = self._param_key_to_shard_info.get(key, None)
        return shard_info


class LayerzeroCheckpoint(object):
    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir
        self.file_list = self._get_files_by_key(ckpt_dir, MODEL_FILE_KEY)
        self.global_state = {}
        self._build_global_state()
        self.state_dicts = [ShardStateDict(f) for f in self.file_list]
        self.pp_degree = self.state_dicts[0].pp_degree
        self.tp_degree = self.state_dicts[0].tp_degree
        self.layer_state_dicts = [{} for _ in range(self.num_layers)]
        self.pre_process_sd = {}
        self.post_process_sd = {}
        self.other_sd = {}
        self._sanity_check()
        self.convert_to_full_state_dict()

    def _sanity_check(self):
        pass

    def _build_global_state(self):
        sd = torch.load(self.file_list[0], map_location=torch.device('cpu'))
        self.global_state[ITERATION_KEY] = sd.get(ITERATION_KEY, 0)
        self.global_state[ARGS_KEY] = sd.get(ARGS_KEY, None)
        args = self.get_args()
        self.global_state[NUM_LAYERS_KEY] = args.num_layers
        self.global_state[PP_LAYERS_KEY] = args.num_layers // args.pipeline_model_parallel_size

    @property
    def pp_layers_per_rank(self):
        return self.global_state[PP_LAYERS_KEY]

    @property
    def num_layers(self):
        return self.global_state[NUM_LAYERS_KEY]

    def get_iteration(self):
        if ITERATION_KEY not in self.global_state:
            sd = torch.load(
                self.mp_rank_files[0], map_location=torch.device('cpu'))
            self.global_state[ITERATION_KEY] = sd.get(ITERATION_KEY, 0)

        return self.global_state[ITERATION_KEY]

    def get_args(self):
        if ARGS_KEY not in self.global_state:
            sd = torch.load(
                self.mp_rank_files[0], map_location=torch.device('cpu'))
            self.global_state[ARGS_KEY] = sd.get(ARGS_KEY, None)

        return self.global_state[ARGS_KEY]

    def _get_files_by_key(self, ckpt_dir, key):
        file_list = []
        for root, dirs, files in os.walk(ckpt_dir):
            for file in files:
                if file.startswith(key):
                    file_list.append(os.path.join(root, file))
        return file_list

    def convert_to_full_state_dict(self) -> Dict[str, Any]:
        state_dicts: List[ShardStateDict] = self.state_dicts
        same_pp_groups = _get_same_pp_ranks(state_dicts)
        for pp_rank, pp_groups in same_pp_groups.items():
            self.build_layer_state_dict(pp_rank, pp_groups)
        return

    def build_layer_state_dict(self, pp_rank: int, state_dicts: List[ShardStateDict]) -> Dict:
        '''
        This function converts dist layerzero state_dict file for each pp model

        Input: sorted state_dict based on global rank and belongs to same pp stage

        output: A single full_state_dict for this pp stage. (TP=1)
        '''
        tp_zero_index = get_TP_unshard_idx_same_pp(state_dicts)
        non_zero_keys = set()
        for key, param in state_dicts[0].model_state_dict.items():
            fqn = clean_tensor_name(key)
            shard_info = state_dicts[0]._get_shard_info_by_fqn(fqn)
            if shard_info is None:
                full_tensor = param
                non_zero_keys.add(fqn)
            else:
                shape = shard_info.shape
                tensor_model_parallel = shard_info.tensor_model_parallel
                partition_dim = shard_info.partition_dim

                shard_lists = _get_shard_list_by_param_key(state_dicts, key)
                if self.tp_degree > 1 and tensor_model_parallel:
                    full_tensor = zero_tp_to_full_tensor(
                        shard_lists, tp_zero_index, shape, partition_dim, self.tp_degree)
                else:
                    full_tensor = zero_to_full_tensor(shard_lists, shape)
            layer_num = _get_layer_num(fqn)
            if layer_num is not None:
                global_layer_num = self.local_to_global_layer_num(
                    layer_num, pp_rank)
                self.layer_state_dicts[global_layer_num][key] = full_tensor
            else:
                if pp_rank == 0:
                    self.pre_process_sd[fqn] = full_tensor
                if pp_rank == self.pp_degree - 1:
                    self.post_process_sd[fqn] = full_tensor
                if not (pp_rank == 0) or (pp_rank == self.pp_degree - 1):
                    self.other_sd[fqn] = full_tensor
        print(f"{non_zero_keys=}")
        return

    def local_to_global_layer_num(self, layer_num: int, pp_rank: int):
        return layer_num + pp_rank * self.pp_layers_per_rank

    def create_rank_checkpoint(self, tp_index: int, pp_index: int, tp_degree: int, pp_degree: int) -> Dict[str, torch.Tensor]:
        '''
        为指定的 tp_index 和 pp_index 生成对应的状态字典，并根据 tp_degree 对张量进行分片。

        Args:
            tp_index (int): 目标 TP 阶段的索引。
            pp_index (int): 目标 PP 阶段的索引。
            tp_degree (int): TP 的总阶段数。
            pp_degree (int): PP 的总阶段数。

        Returns:
            Dict[str, torch.Tensor]: 目标 TP 和 PP 阶段的状态字典。
        '''
        # 获取目标 PP 阶段的状态字典
        state_dict = self.get_layer_state_dict(pp_index, pp_degree)
        # 对状态字典中的张量进行 TP 分片
        rank_state_dict = {}
        for fqn, tensor in state_dict.items():
            shard_info = self.state_dicts[0]._get_shard_info_by_fqn(fqn)
            
            if MODEL_PREFIX:
                fqn = clean_prefix(fqn, MODEL_PREFIX)
                
            if shard_info is not None and shard_info.tensor_model_parallel:
                # 如果张量是 TP 分片的，则根据 tp_index 和 tp_degree 进行分片
                partition_dim = shard_info.partition_dim
                stride = shard_info.partition_stride
                rank_state_dict[fqn] = shard_tensor(
                    tensor, tp_degree, tp_index, partition_dim, stride)
            else:
                # 如果张量不是 TP 分片的，则直接使用原张量
                rank_state_dict[fqn] = tensor
        return rank_state_dict

    def get_layer_state_dict(self, pp_index: int, pp_degree: int) -> Dict[str, torch.Tensor]:
        '''
        获取指定 pp_index 的状态字典，包括预处理、后处理以及该 pp_index 对应的层状态字典。

        Args:
            pp_index (int): 目标 PP 阶段的索引。
            pp_degree (int): PP 的总阶段数。

        Returns:
            Dict[str, torch.Tensor]: 目标 PP 阶段的状态字典。
        '''
        state_dict = {}

        # 添加预处理部分（仅在 pp_index == 0 时）
        if pp_index == 0:
            state_dict.update(self.pre_process_sd)

        # 添加后处理部分（仅在 pp_index == pp_degree - 1 时）
        if pp_index == pp_degree - 1:
            state_dict.update(self.post_process_sd)
        state_dict.update(self.other_sd)
        pp_layers_per_rank = self.pp_layers_per_rank
        # 添加该 PP 阶段对应的层状态字典
        start_layer = pp_index * pp_layers_per_rank
        end_layer = start_layer + pp_layers_per_rank

        for layer_idx, layer_state_dict in enumerate(self.layer_state_dicts[start_layer:end_layer]):
            layer_state_dict = _rename_layer_sd_key(
                layer_state_dict, layer_idx)
            state_dict.update(layer_state_dict)

        return state_dict


def _get_layer_num(key: str) -> int:
    match = PP_LAYER_PATTERN.match(key)

    if match:
        # 提取前缀、层号和后缀
        prefix, layer_num, suffix = match.groups()
        # 构建新的键
        return int(layer_num)
    else:
        return None


def _rename_layer_sd_key(layer_state_dict: Dict, layer_idx: int):
    state_dict = {}
    for key, value in layer_state_dict.items():
        state_dict[_rename_layer_key(key, layer_idx)] = value
    return state_dict


def _rename_layer_key(old_key: str, idx: int) -> str:
    """Generate new key based for pp stage,  old_key -> new_key

    Args:
        old_key (str): layers.{i}.name
        idx (int): num_layers_idx new

    Returns:
        str: layers.{idx}.name
    """
    match = PP_LAYER_PATTERN.match(old_key)

    if match:
        # 提取前缀、层号和后缀
        prefix, layer_num, suffix = match.groups()
        # 构建新的键
        new_key = f"{prefix}{idx}{suffix}"
        return new_key
    else:
        return old_key


def _get_shard_list_by_param_key(state_dicts, key):
    '''
    Return the sharded paramter that belongs to same param key!!!

    Be aware of TP condition, the parameter is shard by TP then by ZeRO3
    '''
    if not state_dicts:
        return []
    resutls = [sd._get_param_by_param_key(key) for sd in state_dicts]
    return resutls


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def shard_tensor(full_tensor: torch.tensor,
                 tp_degree: int,
                 tp_rank: int,
                 partition_dim: int, stride: int = 1
                 ) -> List[torch.tensor]:
    shards = torch.chunk(full_tensor, tp_degree, dim=partition_dim)
    set_tensor_model_parallel_attributes(
        shards[tp_rank], is_parallel=True, dim=partition_dim, stride=stride)
    return shards[tp_rank]


def zero_to_full_tensor(shards, global_shape):
    if not isinstance(global_shape, torch.Size):
        raise TypeError(f"Expect Type torch.Size, got {type(global_shape)}")
    if not all(len(param.shape) <= 1 for param in shards):
        raise AssertionError(f"Expect all zero param to be 1D, Got non Flat param")
    return torch.cat(shards).reshape(global_shape)


def tp_full_shape(shape: torch.Size, partition_dim: int, tp_degree: int):
    if len(shape) <= partition_dim:
        raise AssertionError(f"{partition_dim} greater or equal to shape len {len(shape)}")
    shape_list = list(shape)
    # 修改指定维度的大小
    shape_list[partition_dim] *= tp_degree 
    return torch.Size(shape_list)


def zero_tp_to_full_tensor(shards: List[torch.tensor],
                           tp_zero_index: List[int],
                           shape: torch.Size,
                           partition_dim: int,
                           tp_degree: int):
    if tp_degree > 1:
        if len(shards) != len(tp_zero_index):
            raise AssertionError(f"Not enough zero params for {tp_degree=}")
        full_shape = tp_full_shape(shape, partition_dim, tp_degree)
        shards = [shards[i] for i in tp_zero_index]
    else:
        full_shape = shape
    return zero_to_full_tensor(shards, full_shape)


def _get_same_pp_ranks(shard_dict_list: List[ShardStateDict]) -> Dict[int, List[ShardStateDict]]:
    results = defaultdict(list)
    for shard_dict in shard_dict_list:
        pp_rank = shard_dict.pp_rank
        results[pp_rank].append(shard_dict)

    # 对每组进行 sanity check 和排序
    for pp_rank, group in results.items():
        # 检查所有状态字典是否具有相同的模型键
        model_keys = [set(sd.model_state_dict.keys()) for sd in group]
        if not all(keys == model_keys[0] for keys in model_keys):
            raise ValueError(
                f"All state dicts in PP rank {pp_rank} must have the same model keys. "
                f"Found mismatched keys: {model_keys}"
            )
        # 按全局rank排序排序
        sort_shard_dict_by_global_rank(group)
    return results


def sort_shard_dict_by_global_rank(shard_list: List[ShardStateDict]) -> None:
    shard_list.sort()


def get_TP_unshard_idx_same_pp(state_dicts: List[ShardStateDict]) -> List[int]:
    pp_ranks = set(sd.pp_rank for sd in state_dicts)
    if len(pp_ranks) != 1:
        raise AssertionError("Got more than 1 pp rank")

    tp_global_index = [(idx, sd.tp_rank, sd.global_rank)
                       for idx, sd in enumerate(state_dicts)]
    sorted_list = sorted(tp_global_index, key=lambda x: (x[1], x[2]))
    sorted_index = [x[0] for x in sorted_list]
    return sorted_index
