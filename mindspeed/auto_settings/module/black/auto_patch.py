# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import os
import stat
import time
import json
import re
import atexit
import torch

from mindspeed.auto_settings.utils.logger import get_logger
from mindspeed.auto_settings.utils.singleton import Singleton
from mindspeed.auto_settings.utils.utils import standardize_path


class AutoPatcher(metaclass=Singleton):
    def __init__(self, save_path):
        self.module_profiling_step = 5
        self.stop_profiling_step = 10
        self.curr_step = 0
        self.unit_gb = 1024 ** 3
        self.context = {}
        self.handles = []
        # name format in mcore
        self.profile_modules = ('embedding', '0', 'final_layernorm', 'output_layer')
        self.save_path = standardize_path(save_path, check_read=False)
        atexit.register(self.export_to_file)
        self.logger = get_logger("AutoPatcher")

    def export_to_file(self):
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank == 0:
            self.logger.info(f"rank: {torch.distributed.get_rank()} saving context: {self.context}")
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            modes = stat.S_IWUSR | stat.S_IRUSR
            if os.path.exists(self.save_path):
                self.logger.warning(f'{self.save_path} will be overwrited !')
            with open(self.save_path, 'w') as fout:
                fout.write(json.dumps(self.context))
                os.chmod(self.save_path, 0o640)

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
            self.logger.info(f"hook next_name: {next_name}")

            match_ret = re.search(r'[^.]+$', next_name)
            if match_ret and match_ret.group(0) in self.profile_modules:
                self.handles.append(module.register_forward_pre_hook(self.forward_pre_hook(name)))
                self.handles.append(module.register_forward_hook(self.forward_post_hook(name)))
                continue
            self.register_recursive_hook(next_name, module, ctx)