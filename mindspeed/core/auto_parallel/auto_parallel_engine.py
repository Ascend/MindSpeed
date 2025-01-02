# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import time
import copy
from typing import List

import torch

from .utils import logger
from .utils.parallel_config import ParallelConfig
from .utils.system_config import get_system_config
from .utils.model_config import get_model_config
from .utils.data_type import TaskType
from .utils.data_type import SearchAlgo
from .utils.profiler_args import OperatorProfileArgs
from .utils.profiler import Launch
from .utils.profiler import CommProfiling
from .search.memory_model import MemoryCostModel
from .search.optimizer import Search
from .search.optimizer import SearchByFullPrecision
from .search.optimizer import SearchByFastMode


class SearchEngine:

    search_spaces: List[ParallelConfig] = []

    @classmethod
    def init_global_group(cls):
        from .utils.utils import KVStore
        KVStore.init()

        sys_config = get_system_config()
        init_method = 'tcp://{}:{}'.format(sys_config.master_addr, sys_config.master_port + 1)
        torch.distributed.init_process_group(
            backend=torch.distributed.Backend.GLOO,
            init_method=init_method,
            rank=sys_config.node_rank,
            world_size=sys_config.nnodes
        )

    @classmethod
    def notice_exit_search(cls):
        message = torch.tensor(
            [TaskType.EXIT_SEARCH.value for _ in range(get_system_config().search_dimensions + 1)], 
            dtype=torch.int
        )
        torch.distributed.broadcast(message, src=0)

    @classmethod
    def check_device_sufficient(cls):
        real_world_size = 0
        for config in cls.search_spaces:
            min_dp = (1 if config.expert_model_parallel_size == 0 else config.expert_model_parallel_size)
            real_world_size = max(
                real_world_size,
                config.tensor_model_parallel_size * config.ring_attention_size * config.ulysses_size * min_dp
            )
        return real_world_size <= get_system_config().world_size, real_world_size

    @classmethod
    def build_search_spaces(cls):
        args = get_model_config().args
        sys_config = get_system_config()
        nproc_per_node = sys_config.nproc_per_node
        world_size = sys_config.target_world_size

        for pp in range(1, world_size + 1):
            if world_size % pp != 0 or args.num_layers % pp != 0:
                continue

            for i in range(nproc_per_node):
                tp = 2 ** i
                if tp > nproc_per_node or tp > (world_size // pp):
                    break

                if (args.group_query_attention and args.num_query_groups % tp != 0) \
                    or (args.num_attention_heads % tp != 0):
                    continue

                max_cp_size = world_size // (pp * tp)
                for cp_size in range(1, max_cp_size + 1):
                    if world_size % (pp * tp * cp_size) != 0 or \
                        args.global_batch_size % (world_size // (pp * tp * cp_size)) != 0:
                        continue

                    for up in range(1, cp_size + 1):
                        if cp_size % up != 0:
                            continue

                        cp = cp_size // up

                        head, remainder = divmod(args.num_attention_heads, up * tp)

                        is_ulysses_invalid = (cp == 1 and up > 1) and (head < 1 or remainder != 0)
                        if is_ulysses_invalid:
                            continue

                        if (cp > 1 and up == 1) and (args.seq_length % (2 * cp)) != 0:
                            continue

                        is_hybrid_algo_invalid = (cp > 1 and up > 1) and ((head < 1 or remainder != 0) or
                                                                          (args.seq_length % (2 * cp) != 0))
                        if is_hybrid_algo_invalid:
                            continue

                        dp = world_size // (pp * tp * cp_size)
                        dp_batch_size = args.global_batch_size // dp
                        for num_microbatch in range(1, dp_batch_size + 1):
                            if dp_batch_size % num_microbatch != 0:
                                continue

                            mbs = dp_batch_size // num_microbatch
                            cls.search_spaces.append(ParallelConfig(
                                pp, tp, dp, cp, up, mbs, 1, 0 if args.num_experts is None else 1
                            ))

        # vpp维度
        temp_search_spaces: List[ParallelConfig] = copy.deepcopy(cls.search_spaces)
        for config in temp_search_spaces:
            pp, tp, dp, cp, up, mbs, _, _ = config.to_list()
            num_microbatch = args.global_batch_size // dp // mbs

            if pp > 2 and num_microbatch % pp == 0:
                num_layers_per_pp_stage = args.num_layers // pp
                for vpp in range(2, num_layers_per_pp_stage + 1):
                    if num_layers_per_pp_stage % vpp != 0:
                        continue
                    cls.search_spaces.append(ParallelConfig(
                        pp, tp, dp, cp, up, mbs, vpp, 0 if args.num_experts is None else 1
                    ))
        
        # ep维度
        if args.num_experts:
            temp_search_spaces = copy.deepcopy(cls.search_spaces)
            for config in temp_search_spaces:
                pp, tp, dp, cp, up, mbs, vpp, _ = config.to_list()
                for ep in range(2, dp + 1):
                    if args.num_experts % ep != 0 or dp % ep != 0 or (dp * cp * up) % ep != 0:
                        continue
                    cls.search_spaces.append(ParallelConfig(pp, tp, dp, cp, up, mbs, vpp, ep))

        # filter unvalid configs
        temp_search_spaces = copy.deepcopy(cls.search_spaces)
        cls.search_spaces.clear()
        for config in temp_search_spaces:
            pp, tp, dp, cp, up, mbs, vpp, ep = config.to_list()
            num_microbatch = config.num_microbatch
            splited_seq_len = config.splited_seq_len
            cond = [
                mbs in (1, 2),
                splited_seq_len >= 4 * 1024 if cp * up > 1 else True
            ]
            if all(cond):
                SearchEngine.search_spaces.append(config)

    @classmethod
    def search_on_slave(cls):
        if get_system_config().search_algo == SearchAlgo.FAST_MODE.value:
            len_search_space = torch.tensor([0], dtype=torch.int)
            torch.distributed.broadcast(len_search_space, src=0)
            len_search_space = len_search_space.item()
            logger.info(f"length of search space: {len_search_space}")
            
            search_space = torch.empty([len_search_space, get_system_config().search_dimensions], dtype=torch.int)
            torch.distributed.broadcast(search_space, src=0)
            for config in search_space:
                config = ParallelConfig.from_tensor(config)
                cls.search_spaces.append(config)
                logger.info(f"recv config: {config}")
            
            CommProfiling.profiler_comm_times(cls.search_spaces)

        while True:
            try:
                time.sleep(1)
                logger.info("wait next task...")
                message = torch.tensor([0 for _ in range(get_system_config().search_dimensions + 1)], dtype=torch.int)
                torch.distributed.broadcast(message, src=0)

                task_type = message[-1].item()
                config = ParallelConfig.from_tensor(message)
                if task_type == TaskType.EXIT_SEARCH:
                    break
                Launch.launch(OperatorProfileArgs(config), task_type)
            except BaseException as e:
                logger.error("wait next task timeout!!!")

    @classmethod
    def search_on_master(cls):
        search_space = []
        for config in cls.search_spaces:
            pmem = MemoryCostModel.get_peak_memory(config, 'white_box')
            if pmem <= get_system_config().max_available_memory * 1.2:
                search_space.append(config)
        cls.search_spaces = copy.deepcopy(search_space)
        Search.print_search_space('Filterd search spaces', cls.search_spaces)

        if get_system_config().search_algo == SearchAlgo.FULL_PRECISION.value:
            result = SearchByFullPrecision().search(cls.search_spaces)
        else:
            torch.distributed.broadcast(torch.tensor([len(cls.search_spaces)], dtype=torch.int), src=0)

            search_space = [config.to_list() for config in cls.search_spaces]
            torch.distributed.broadcast(torch.tensor(search_space, dtype=torch.int), src=0)

            CommProfiling.profiler_comm_times(cls.search_spaces)

            result = SearchByFastMode().search(cls.search_spaces)

        cls.notice_exit_search()
        return result

    @classmethod
    def search(cls):
        # 非主节点不走搜索流程
        if get_system_config().node_rank != 0:
            cls.search_on_slave()
            return

        cls.build_search_spaces()
        is_suff, world_size = cls.check_device_sufficient()
        if not is_suff:
            logger.fatal(f"### Need {world_size} devices for search.")
            cls.notice_exit_search()
            return
        
        Search.print_search_space('Full search spaces', cls.search_spaces)
        result = cls.search_on_master()

        logger.info(f"{'#' * 10} search results:")
        for i, _ in enumerate(result):
            config = result[i]
            logger.info(f"### config-{i}:")
            logger.info(f'   --pipeline-model-parallel-size: {config.pipeline_model_parallel_size}')
            logger.info(f'   --tensor-model-parallel-size: {config.tensor_model_parallel_size}')
            logger.info(f'   --data-parallel-size: {config.data_parallel_size}')

            context_parallel_size = config.ring_attention_size * config.ulysses_size
            context_parallel_algo = 'hybrid_cp_algo'
            if context_parallel_size == config.ring_attention_size:
                context_parallel_algo = 'context_cp_algo'
            if context_parallel_size == config.ulysses_size:
                context_parallel_algo = 'ulysses_cp_algo'
            
            if context_parallel_size > 1:
                logger.info(f'   --context-parallel-size: {context_parallel_size}')
                logger.info(f'   --context-parallel-algo: {context_parallel_algo}')

            logger.info(f'   --micro-batch-size: {config.micro_batch_size}')

            if config.virtual_pipeline_model_parallel_size > 1:
                logger.info(f'   --virtual-pipeline-model-parallel-size: {config.virtual_pipeline_model_parallel_size}')
                logger.info(f'   --num-layers-per-virtual-pipeline-stage: {config.num_layers_per_virtual_pipeline_stage}')

            if config.expert_model_parallel_size > 0:
                logger.info(f'   --expert-model-parallel-size: {config.expert_model_parallel_size}')

        



        
        

                            

            


                
            

        