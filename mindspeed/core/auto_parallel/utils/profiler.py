# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import os
import sys
import stat
import math
import time
import json
import re
import operator
import atexit
import functools
import subprocess
import signal
import threading
from typing import List

import pandas as pd
import torch

from . import logger
from .system_config import get_system_config
from .model_config import get_model_config
from .parallel_config import ParallelConfig
from .profiler_args import Arguments
from .data_type import TaskType
from .utils import KVStore
from .utils import SingletonType
from .utils import get_cache_path
from .utils import GlobalMemoryBuffer

    
class Launch:
    @staticmethod
    def monitor_next(process):
        while True:
            exit_flag = KVStore.get("exit")
            if int(exit_flag) == 1:
                try:
                    process_group_id = os.getpgid(process.pid)
                    os.killpg(process_group_id, signal.SIGKILL)
                    break
                except ProcessLookupError:
                    break
            time.sleep(60)
    
    @staticmethod
    def launch(args: Arguments, task_type: TaskType):
        def help_launch(args: Arguments):
            sys_config = get_system_config()
            command = [
                'torchrun',
                '--nproc_per_node', str(sys_config.nproc_per_node),
                '--nnodes', str(sys_config.nnodes),
                '--node-rank', str(sys_config.node_rank),
                '--master_addr', str(sys_config.master_addr),
                '--master_port', str(sys_config.master_port - 1),
                str(sys.argv[0])
            ] + args.to_list()

            log_str = 'start launch: \n'
            log_str += command[0]
            for i in range(1, len(command)):
                if command[i].startswith('--'):
                    log_str += '\n'
                    log_str += command[i]
                else:
                    log_str += ' '
                    log_str += command[i]
            log_str += '\n'
            logger.info(log_str)

            KVStore.set("exit", "0")
            process = subprocess.Popen(command, shell=False, preexec_fn=lambda: os.setpgrp())
            monitor_thread = threading.Thread(target=Launch.monitor_next, args=(process,))
            monitor_thread.start()
            process.wait()
            KVStore.set("exit", "1")
            torch.distributed.barrier()

        if get_system_config().node_rank != 0:
            help_launch(args)
            return
        
        if os.path.exists(args.config.module_profile_path(node_rank=0)):
            return
        
        buffer = args.config.to_list() + [task_type.value]
        torch.distributed.broadcast(torch.tensor(buffer, dtype=torch.int), 0)

        help_launch(args)


class AutoProfiler(metaclass=SingletonType):
    def __init__(self, save_path):
        self.module_profiling_step = 5
        self.stop_profiling_step = 10
        self.curr_step = 0
        self.unit_gb = 1024**3
        self.context = {}
        self.handles = []
        # name format in mcore
        self.profile_modules = ('embedding', '0', 'final_layernorm', 'output_layer')
        self.save_path = save_path
        atexit.register(self.export_to_file)
    
    def export_to_file(self):
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank == 0:
            logger.info(f"rank: {torch.distributed.get_rank()} saving context: {self.context}")
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(self.save_path, flags, modes), 'w') as fout:
                fout.write(json.dumps(self.context))
                fout.flush()

    @staticmethod
    def get_memory_status():
        memory = torch.npu.memory_allocated()
        max_memory = torch.npu.max_memory_allocated()
        return memory, max_memory

    def should_profiling(self, collect_step_time=False):
        # 分为两个阶段，避免采集module profiling数据时插入的synchronize影响单步耗时的精度
        if collect_step_time:
            return self.module_profiling_step <= self.curr_step < self.stop_profiling_step
        else:
            return self.curr_step < self.module_profiling_step

    def hook_train_step(self, train_step):
        def custom_train_step(*args, **kwargs):
            # 在采集单步耗时前需要移除hook函数
            if self.should_profiling(collect_step_time=True):
                for handle in self.handles:
                    handle.remove()
            # 采集单步耗时数据
            torch.cuda.synchronize()
            start_time = time.time()
            result = train_step(*args, **kwargs)
            torch.cuda.synchronize()
            step_time = time.time() - start_time
            if self.should_profiling(collect_step_time=True):
                cur_step_time = self.context.get('step_time', 0)
                cur_step_time = (cur_step_time * (self.curr_step - self.module_profiling_step) + step_time) \
                    / (self.curr_step - self.module_profiling_step + 1)
                self.context['step_time'] = cur_step_time
            self.curr_step += 1
            return result
        return custom_train_step

    def forward_pre_hook(self, module_name):
        if module_name not in self.context.keys():
            self.context[module_name] = dict()

        def hook(module, *args, **kargs):
            if self.should_profiling(collect_step_time=False):
                if module_name not in self.context:
                    self.context[module_name] = {}

                torch.npu.synchronize()
                mem, _ = self.get_memory_status()
                self.context[module_name]['time'] = time.time()
                self.context[module_name]['memory'] = mem
                self.context[module_name]['max_memory'] = mem
                torch.npu.reset_max_memory_allocated()

        return hook

    def forward_post_hook(self, module_name):
        def hook(module, *args, **kargs):
            if self.should_profiling(collect_step_time=False):
                torch.npu.synchronize()
                self.context[module_name]['time'] = (time.time() - self.context[module_name]['time']) * 1000
                mem, max_mem = self.get_memory_status()
                mem1, mem2 = self.context[module_name]['memory'], self.context[module_name]['max_memory']
                self.context[module_name]['memory'] = (mem - mem1) / self.unit_gb
                self.context[module_name]['max_memory'] = (max_mem - mem2) / self.unit_gb

        return hook

    def register_recursive_hook(self, prefix_name, model, ctx):
        model = model[0] if isinstance(model, list) else model
        for name, module in model.named_children():
            next_name = prefix_name + "." + name if prefix_name != "" else name
            logger.info(f"hook next_name: {next_name}")

            match_ret = re.search(r'[^.]+$', next_name)
            if match_ret and match_ret.group(0) in self.profile_modules:
                self.handles.append(module.register_forward_pre_hook(self.forward_pre_hook(name)))
                self.handles.append(module.register_forward_hook(self.forward_post_hook(name)))
                continue
            self.register_recursive_hook(next_name, module, ctx)


class CommProfiling:
    band_width_undirectional = 25
    cache = None
    cache_path = None

    @classmethod
    def mk_cache(cls):
        cls.cache_path = get_cache_path() + os.sep + 'comm_prof.csv'
        if os.path.exists(cls.cache_path):
            cls.cache = pd.read_csv(cls.cache_path)
        else:
            cls.cache = pd.DataFrame(columns=['type', 'domains', 'datasize', 'time'])

    @classmethod
    def get_comm_time(cls, shape, domains, op):
        if domains == 1:
            return 0

        data_size = cls.cal_data_size(shape)
        data = cls.cache[
            (cls.cache['type'] == op) &
            (cls.cache['datasize'] == data_size) &
            (cls.cache['domains'] == domains)
        ]

        return data['time'].mean() if len(data['time'].index) > 0 else None

    @staticmethod
    def get_all_reduce(*args):
        input_ = args[0]
        torch.distributed.all_reduce(input_)
        return input_

    @staticmethod
    def get_all_gather(*args):
        input_, domain = args
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * domain
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        torch.distributed._all_gather_base(
            output,
            input_.contiguous()
        )
        return output

    @staticmethod
    def get_reduce_scatter(*args):
        input_, domain = args
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] // domain
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        torch.distributed._reduce_scatter_base(
            output,
            input_.contiguous()
        )
        return output

    @staticmethod
    def get_alltoall(*args):
        input_, domain = args
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * domain
        dim_size[2] = dim_size[2] // domain
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        torch.distributed.all_to_all_single(
            output,
            input_.contiguous()
        )
        return output

    @staticmethod
    def call_func(op_type, tensor_shape, domains):
        func_table = {
            'all_reduce': CommProfiling.get_all_reduce,
            'all_gather': CommProfiling.get_all_gather,
            'reduce_scatter': CommProfiling.get_reduce_scatter,
            'alltoall': CommProfiling.get_alltoall
        }

        if op_type not in func_table.keys():
            raise AssertionError(f"can't find {op_type} in func_table")

        data = GlobalMemoryBuffer.get_tensor(tensor_shape, 0)
        comm_times = []
        for _ in range(5):
            torch.distributed.barrier()
            torch.npu.synchronize()
            start_time = time.time()
            func_table[op_type](data, domains)
            torch.npu.synchronize()
            comm_times.append(time.time() - start_time)

        comm_times.remove(max(comm_times))
        comm_times.remove(min(comm_times))
        comm_time = (sum(comm_times) / len(comm_times)) * 1e6
        return comm_time

    @staticmethod
    def comm_eval(
        rank, world_size, distributed_args, commands, comm_times
    ):
        nproc_per_node = distributed_args[0]
        node_rank = distributed_args[1]
        master_addr = distributed_args[2]
        master_port = distributed_args[3]

        torch.npu.set_device(rank)
        global_rank = node_rank * nproc_per_node + rank
        init_method = 'tcp://{}:{}'.format(master_addr, master_port)
        torch.distributed.init_process_group(
            backend=torch.distributed.Backend.NCCL,
            init_method=init_method,
            rank=global_rank,
            world_size=world_size
        )

        for cmd in commands:
            comm_time = CommProfiling.call_func(cmd[0], cmd[-1], cmd[1])
            if rank in (0,):
                comm_times.append([cmd[0], cmd[1], cmd[2], comm_time])

    @classmethod
    def start_eval(cls, group_hccl_commands):
        sys_config = get_system_config()
        manager = torch.multiprocessing.Manager()
        for key, value in group_hccl_commands.items():
            world_size = key
            nprocs = world_size if world_size < sys_config.nproc_per_node else sys_config.nproc_per_node
            need_launch_node = math.ceil(world_size / sys_config.nproc_per_node) - 1

            if sys_config.node_rank <= need_launch_node:
                comm_times = manager.list()
                distributed_args = (sys_config.nproc_per_node, sys_config.node_rank, sys_config.master_addr,
                                    sys_config.master_port - 1)
                torch.multiprocessing.spawn(
                    CommProfiling.comm_eval,
                    args=(world_size,
                          distributed_args,
                          value,
                          comm_times),
                    nprocs=nprocs,
                )
                if len(comm_times) != len(value):
                    logger.error(f"comm_prof error, cur_domains is {world_size}")

                for idx in range(len(value)):
                    cls.cache.loc[len(cls.cache.index)] = (
                        comm_times[idx][0],
                        comm_times[idx][1],
                        cls.cal_data_size(comm_times[idx][2]),
                        comm_times[idx][-1]
                    )
            
            logger.info(f"domains({world_size}) completed, cache: {cls.cache}")

    @staticmethod
    def get_send_recv_time(shape):
        data_size = functools.reduce(operator.mul, shape) * 2 / (1024 ** 3) # GB
        return (data_size / CommProfiling.band_width_undirectional) * 1e6

    @staticmethod
    def cal_data_size(shape):
        return functools.reduce(operator.mul, shape) * 2 # Byte

    @classmethod
    def profiler_comm_times(cls, search_spaces: List[ParallelConfig]):
        cls.mk_cache()

        hccl_command = []
        args = get_model_config().args
        for config in search_spaces:
            dp = config.data_parallel_size
            tp = config.tensor_model_parallel_size
            cp = config.ring_attention_size
            up = config.ulysses_size
            mbs = config.micro_batch_size

            # tp
            if not get_model_config().args.sequence_parallel:
                shape = [args.seq_length, mbs, args.hidden_size]
                command = ['all_reduce', tp, shape]
                if tp != 1 and command not in hccl_command and not cls.get_comm_time(shape, tp, 'all_reduce'):
                    hccl_command.append(command)
            else:
                shape = [args.seq_length // tp // cp // up, mbs, args.hidden_size]
                command = ['all_gather', tp, shape]
                if tp != 1 and command not in hccl_command and not cls.get_comm_time(shape, tp, 'all_gather'):
                    hccl_command.append(command)

                shape = [args.seq_length // cp // up, mbs, args.hidden_size]
                command = ['reduce_scatter', tp, shape]
                if tp != 1 and command not in hccl_command and not cls.get_comm_time(shape, tp, 'reduce_scatter'):
                    hccl_command.append(command)

            # up
            shape = [
                    args.seq_length // cp // up, mbs,
                    args.num_attention_heads // tp,
                    args.hidden_size // args.num_attention_heads
                ]
            command = ['alltoall', up, shape]
            if up != 1 and command not in hccl_command and not cls.get_comm_time(shape, up, 'alltoall'):
                hccl_command.append(command)
            
        hccl_command_dict = {}
        for command in hccl_command:
            domains = command[1]
            if domains not in hccl_command_dict.keys():
                hccl_command_dict[domains] = []
            hccl_command_dict[domains].append(command)

        hccl_command_dict_sorted = dict(
            sorted(hccl_command_dict.items(), key=lambda item: item[0], reverse=True)
        )

        cls.start_eval(hccl_command_dict_sorted)
        logger.info(f"comm cache: {CommProfiling.cache}")

        if get_system_config().node_rank == 0:
            cls.cache.to_csv(cls.cache_path, index=False)