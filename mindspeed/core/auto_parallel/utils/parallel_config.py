# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import os
from dataclasses import dataclass, field

import torch

from .model_config import get_model_config
from .system_config import get_system_config
from .utils import get_cache_path


@dataclass
class ParallelConfig:
    pipeline_model_parallel_size: int

    tensor_model_parallel_size: int

    data_parallel_size: int

    ring_attention_size: int

    ulysses_size: int

    micro_batch_size: int

    virtual_pipeline_model_parallel_size: int

    expert_model_parallel_size: int

    num_layers_per_virtual_pipeline_stage: int = field(init=0)

    paras_memory: float = field(init=.0)

    optim_memory: float = field(init=.0)

    act_memory: float = field(init=.0)

    peak_memory: float = field(init=.0)

    iteration_time: float = field(init=.0)

    def __post_init__(self):
        self.num_layers_per_virtual_pipeline_stage = get_model_config().args.num_layers \
            // self.pipeline_model_parallel_size // self.virtual_pipeline_model_parallel_size

    def __repr__(self) -> str:
        return 'pp{} tp{} dp{} cp{} up{} mbs{} vpp{} ep{}'.format(
            self.pipeline_model_parallel_size,
            self.tensor_model_parallel_size,
            self.data_parallel_size,
            self.ring_attention_size,
            self.ulysses_size,
            self.micro_batch_size,
            self.virtual_pipeline_model_parallel_size,
            self.expert_model_parallel_size,
        )
    
    def __eq__(self, other) -> bool:
        return self.pipeline_model_parallel_size == other.pipeline_model_parallel_size and \
            self.tensor_model_parallel_size == other.tensor_model_parallel_size and \
            self.data_parallel_size == other.data_parallel_size and \
            self.ring_attention_size == other.ring_attention_size and \
            self.ulysses_size == other.ulysses_size and \
            self.micro_batch_size == other.micro_batch_size and \
            self.virtual_pipeline_model_parallel_size == other.virtual_pipeline_model_parallel_size and \
            self.expert_model_parallel_size == other.expert_model_parallel_size

    @property
    def num_microbatch(self):
        return get_model_config().args.global_batch_size // self.data_parallel_size // self.micro_batch_size
    
    @property
    def splited_seq_len(self):
        return get_model_config().args.seq_length // (self.ring_attention_size * self.ulysses_size)

    @property
    def bubble_ratio(self):
        pp = self.pipeline_model_parallel_size
        vp = self.virtual_pipeline_model_parallel_size
        return (pp - 1) / self.num_microbatch / vp
    
    @property
    def operator_profile_path(self):
        tmp_config = self.to_list() + [get_system_config().node_rank]
        dir_name = 'PP{}_TP{}_DP{}_CP{}_UP{}_MBS{}_VP{}_EP{}_node{}_OPERATOR'.format(*tmp_config)
        return str(get_cache_path() + os.sep + dir_name)
    
    @staticmethod
    def from_list(config: list):
        return ParallelConfig(*config)

    @staticmethod
    def from_tensor(config: torch.Tensor):
        return ParallelConfig.from_list(
            config[:get_system_config().search_dimensions].tolist()
        )

    def module_profile_path(self, node_rank=None):
        tmp_config = self.to_list() + [node_rank if node_rank else get_system_config().node_rank]
        file_name = 'PP{}_TP{}_DP{}_CP{}_UP{}_MBS{}_VP{}_EP{}_node{}_MODULE.json'.format(*tmp_config)
        return str(get_cache_path() + os.sep + file_name)

    def crop_config(self):
        world_size = get_system_config().world_size
        return ParallelConfig(
            1, 
            self.tensor_model_parallel_size,
            world_size // (self.tensor_model_parallel_size * self.ring_attention_size * self.ulysses_size),
            self.ring_attention_size,
            self.ulysses_size,
            self.micro_batch_size,
            1,
            self.expert_model_parallel_size
            )
    
    def to_list(self):
        return [
            self.pipeline_model_parallel_size,
            self.tensor_model_parallel_size,
            self.data_parallel_size,
            self.ring_attention_size,
            self.ulysses_size,
            self.micro_batch_size,
            self.virtual_pipeline_model_parallel_size,
            self.expert_model_parallel_size
        ]
    
    

    
