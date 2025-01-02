# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import copy
import logging

from megatron.training.global_vars import build_tokenizer

from .utils.system_config import SystemConfig
from .utils.system_config import set_system_config
from .utils.model_config import ModelConfig
from .utils.model_config import set_model_config
from .auto_parallel_engine import SearchEngine


def search_optimal_configuration(args):
    if not args.use_mcore_models:
        raise AssertionError('gohyper only support mcore models temporily.')
    
    if not hasattr(args, 'padded_vocab_size') or args.padded_vocab_size is None:
        build_tokenizer(args)

    if args.num_experts:
        # fast_mode search can not support moe, use full_precision to search
        args.search_algo = 'full_precison'
    
    model_config = ModelConfig(args=copy.deepcopy(args))
    set_model_config(model_config)

    sys_config = SystemConfig(
        nnodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=int(args.master_port),
        target_nnodes=args.target_nnodes,
        search_algo=args.search_algo,
        work_dir=args.work_dir,
        log_level=logging.DEBUG,
        search_dimensions=8
    )
    set_system_config(sys_config)

    # 使用多节点搜索时，需要先建立全局通信域用于广播探索配置
    SearchEngine.init_global_group()
    SearchEngine.search()


