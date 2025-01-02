# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import os
import sys
from abc import ABC, abstractmethod

from .parallel_config import ParallelConfig


class Arguments(ABC):
    def __init__(self, config: ParallelConfig):
        self.args = Arguments.get_original_args()
        self.config = config
        self.set_default_args()
        self.set_parallel_args(config)
        self.set_custom_arguments()
    
    def to_list(self):
        argv: list = []
        for key, value in self.args.items():
            argv.append(key)
            if not isinstance(value, bool):
                argv.append(value)
        return argv

    @staticmethod
    def get_original_args():
        args = dict()
        argv = sys.argv[1:]
        len_argv = len(argv)
        index = 0
        while index < len_argv:
            if argv[index].startswith('--') and \
                (index + 1 < len_argv and not argv[index + 1].startswith('--')):
                args[argv[index]] = argv[index + 1]
                index += 2
            else:
                args[argv[index]] = True
                index += 1
        return args
    
    @abstractmethod
    def set_custom_arguments(self):
        raise NotImplementedError('This method should be overridden by subclasses')

    def rm_arg(self, key):
        if key in self.args:
            self.args.pop(key)

    def set_default_args(self):
        self.args['--sequence-parallel'] = True
        self.args['--eval-iters'] = '0'
        self.args['--train-iters'] = '10'
        self.rm_arg('--auto-parallel')
        self.rm_arg('--noop-layers')
        self.rm_arg('--lr-warmup-iters')
        self.rm_arg('--save')
        self.rm_arg('--load')
        self.rm_arg('--num-layers-per-virtual-pipeline-stage')
        self.rm_arg('--expert-model-parallel-size')

    def set_parallel_args(self, config: ParallelConfig):
        self.args['--pipeline-model-parallel-size'] = str(config.pipeline_model_parallel_size)
        self.args['--tensor-model-parallel-size'] = str(config.tensor_model_parallel_size)
        self.args['--context-parallel-size'] = str(config.ring_attention_size * config.ulysses_size)
        self.args['--ulysses-degree-in-cp'] = str(config.ulysses_size)
        self.args['--micro-batch-size'] = str(config.micro_batch_size)

        if config.virtual_pipeline_model_parallel_size > 1:
            self.args['--num-layers-per-virtual-pipeline-stage'] = str(config.num_layers_per_virtual_pipeline_stage)
        if config.expert_model_parallel_size > 0:
            self.args['--expert-model-parallel-size'] = str(config.expert_model_parallel_size)
        
        if config.ring_attention_size > 1 and config.ulysses_size > 1:
            self.args['--context-parallel-algo'] = 'hybrid_cp_algo'
            self.args['--use-cp-send-recv-overlap'] = True
        elif config.ring_attention_size > 1 and config.ulysses_size == 1:
            self.args['--context-parallel-algo'] = 'megatron_cp_algo'
            self.args['--use-cp-send-recv-overlap'] = True
        elif config.ring_attention_size == 1 and config.ulysses_size > 1:
            self.args['--context-parallel-algo'] = 'ulysses_cp_algo'
        else:
            self.args['--context-parallel-algo'] = 'ulysses_cp_algo'
    
    
class OperatorProfileArgs(Arguments):
    def set_custom_arguments(self):
        cfg = self.config
        pp = cfg.pipeline_model_parallel_size
        self.args['--num-layers'] = str(pp)
        self.args['--global-batch-size'] = str(pp * cfg.data_parallel_size * cfg.micro_batch_size)
        self.args['--prof-file'] = cfg.module_profile_path()
        self.args['--profile'] = True
        self.args['--profile-step-start'] = '2'
        self.args['--profile-step-end'] = '3'
        self.args['--profile-ranks'] = '0'
        self.args['--profile-level'] = 'level1'
        self.args['--profile-save-path'] = cfg.operator_profile_path
        self.args['--profile-with-cpu'] = True
        self.args['--profile-record-shapes'] = True
        self.args['--profile-with-stack'] = True