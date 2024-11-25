import os
import stat

from functools import wraps
from collections.abc import Iterable
from typing import Dict, List
import pickle
import acl
import torch
import torch.nn
from megatron.training.global_vars import get_args

from mindspeed.core.memory.adaptive_recomputing.swap_manager import get_tensor_mem_size


class RecomputeParser:
    recompute_parser = None

    def __init__(self):
        # layer profiling info
        self.context = {
            'module': []
        }
        self.models = None
        # record allowed recomputing module
        self.allowed_recomputing_module = []
        # profiling prefix
        self.profiling_prefix = ""
        # save modules hook, remove it after apply policy
        self.modules_hooks = []
        # current profiling step
        self.profiling_step = 0
        # step skip profiling, default is 3
        self.skip_profiling_step = 3
        # step for stop profiling, default is 6
        self.stop_profiling_step = 6
        # unit for device memory size(MB)
        self.unit_mb = 1024 * 1024
        # store all module event
        '''
        {
            full_name1: [[, ][, ][, ][, ]]
            full_name2: [[, ][, ][, ]]
            full_name3: [[, ][, ]]
        }
        '''
        self.event_dict: Dict[str, List] = {}

    @staticmethod
    def get_memory_status():
        free, all_memory, _ = acl.rt.get_mem_info(1)
        memory_info = {
            "free": free,
            "all_memory": all_memory,
            "used_memory": torch.npu.memory_allocated(),
            "reserved_memory": torch.npu.memory_reserved(),
            "max_memory_allocated": torch.npu.max_memory_allocated()
        }

        return memory_info

    def pre_hook_func(self, state, *args, **kargs):
        if 'memory' not in state:
            state['memory'] = 0
        state['input'] = self.cal_input_output_size(args)
        if self.profiling_step == self.stop_profiling_step:
            state['memory'] = torch.npu.memory_allocated() - state['input'] * self.unit_mb
            print(f"success print pre hook memory = {state['memory']}")
        cur_module_full_name = state['prefix_name'] + '.' + state['name']
        if cur_module_full_name not in self.event_dict.keys():
            self.event_dict[cur_module_full_name] = []
        if self.profiling_step < self.stop_profiling_step:
            start_event = torch.npu.Event(enable_timing=True)
            self.event_dict[cur_module_full_name].append([start_event])
            start_event.record()

    def post_hook_func(self, state, args, output):
        if self.profiling_step < self.stop_profiling_step:
            cur_module_full_name = state['prefix_name'] + '.' + state['name']
            end_event = torch.npu.Event(enable_timing=True)
            end_event.record()
            # add end_event to corresponding position of list
            for item in reversed(self.event_dict[cur_module_full_name]):
                if len(item) == 1:
                    item.append(end_event)
                    break

        if self.profiling_step == self.stop_profiling_step:
            output_memory = self.cal_input_output_size(output)
            state['memory'] = (torch.npu.memory_allocated() - state['memory']) // self.unit_mb
            print(f"success print post hook memory = {state['memory']} and output_memory  = {output_memory}")
            state['input'] += output_memory

    def forward_pre_hook(self, ctx):
        def hook(module, *args, **kargs):
            if 'module' in self.context:
                self.context['module'].append(ctx)
            self.pre_hook_func(ctx, *args, **kargs)

        return hook

    def forward_post_hook(self, ctx):
        def hook(module, args, output):
            self.post_hook_func(ctx, args, output)
            if 'module' in self.context:
                self.context['module'].pop()

        return hook

    def construct_context_recursive(self, prefix_name, model, ctx, have_allowed_recomputing):
        # 1.construct context
        next_have_allowed_recomputing = have_allowed_recomputing
        for name, module in model.named_children():
            if 'layers' not in ctx:
                ctx['layers'] = []

            current_ctx = {'name': name, 'prefix_name': prefix_name}
            if 'layers' in ctx:
                ctx['layers'].append(current_ctx)

            next_name = prefix_name + "." + name if prefix_name != "" else name

            # 2.tag allowed_recomputing module
            if have_allowed_recomputing:
                for allowed_recomputing_module in self.allowed_recomputing_module:
                    if isinstance(module, allowed_recomputing_module):
                        current_ctx['allowed_recomputing'] = True
                        if isinstance(model, torch.nn.ModuleList):
                            ctx['is_module_list'] = True
                            ctx['is_recomputing_layer'] = True
                        else:
                            current_ctx['is_recomputing_layer'] = True
                        next_have_allowed_recomputing = False
            self.construct_context_recursive(next_name, module, current_ctx, next_have_allowed_recomputing)

    def register_recursive_hook(self, model, ctx, profiling_prefix, layer_index=0):
        index = layer_index or 0
        for module in model.children():
            if 'layers' not in ctx:
                continue
            current_ctx = ctx['layers'][index]
            prefix_name = current_ctx['prefix_name']
            name = current_ctx['name']

            is_recomputing_layer = not isinstance(module, torch.nn.ModuleList) and 'is_recomputing_layer' in current_ctx
            is_allowed_recomputing = 'allowed_recomputing' in current_ctx and index == 0
            if is_recomputing_layer or is_allowed_recomputing:
                profiling_prefix = prefix_name + "." + name
                pre_hook = module.register_forward_pre_hook(self.forward_pre_hook(current_ctx))
                post_hook = module.register_forward_hook(self.forward_post_hook(current_ctx))
                self.modules_hooks.append(pre_hook)
                self.modules_hooks.append(post_hook)
            elif profiling_prefix and prefix_name.startswith(profiling_prefix):
                pre_hook = module.register_forward_pre_hook(self.forward_pre_hook(current_ctx))
                post_hook = module.register_forward_hook(self.forward_post_hook(current_ctx))
                self.modules_hooks.append(pre_hook)
                self.modules_hooks.append(post_hook)
            self.register_recursive_hook(module, current_ctx, profiling_prefix)
            index += 1

    def reset_modules(self):
        if torch.distributed.get_rank() % 8 == 0:
            ootb_context_path = get_args().profile_save_path
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            mode = stat.S_IWUSR | stat.S_IRUSR
            ootb_context_path_json = f'{ootb_context_path}.json'
            with os.fdopen(os.open(ootb_context_path_json, flags, mode=mode), "wb") as file:
                file.write(pickle.dumps(self.context))

    def hook_step_func(self, step_func, models):
        def custom_step_func(*args, **kargs):
            result = step_func(*args, **kargs)
            if self.profiling_step >= self.stop_profiling_step + 1:
                return result
            memory_info = self.get_memory_status()
            try:
                self.context['used_mem'] = memory_info["used_memory"] // self.unit_mb
                self.context['max_device_memory'] = memory_info["all_memory"] // self.unit_mb
            except KeyError:
                print("[ERROR] Some of these keys don't exist.")
            self.profiling_step += 1
            torch.npu.synchronize()
            # record module time
            cal_module_forward_time(self.context, self.event_dict)

            # reset modules
            if self.profiling_step == self.stop_profiling_step + 1:
                self.reset_modules()
            return result
        return custom_step_func

    def add_allowed_recomputing_module(self, module):
        if module not in self.allowed_recomputing_module:
            self.allowed_recomputing_module.append(module)
            print(f"after append self.allowed_recomputing_module = {self.allowed_recomputing_module} and module = {module}")

    def cal_input_output_size(self, args):
        size = 0
        if isinstance(args, torch.Tensor):
            size += get_tensor_mem_size(args)
            return size // self.unit_mb
        for arg in args:
            if isinstance(arg, torch.Tensor):
                size += get_tensor_mem_size(arg)
            elif isinstance(arg, Iterable):
                for t in arg:
                    if isinstance(t, torch.Tensor):
                        size += get_tensor_mem_size(t)
                    elif t is None:
                        pass
                    else:
                        print(f"warning: unknown input/output type {str(type(t))}")
            elif arg is None:
                pass
            else:
                print(f"warning: unknown input/output type {str(type(arg))}")
        return size // self.unit_mb


def get_recompute_parser():
    if RecomputeParser.recompute_parser is None:
        RecomputeParser.recompute_parser = RecomputeParser()
    return RecomputeParser.recompute_parser


def setup_model_and_optimizer_decorator(setup_model_and_optimizer):
    @wraps(setup_model_and_optimizer)
    def wrapper(*args, **kargs):
        models, optimizer, opt_param_scheduler = setup_model_and_optimizer(*args, **kargs)
        if os.getenv('OOTB_OPTIMIZER_PROFILING', 'FALSE') != 'TRUE':
            print("OOTB_OPTIMIZER_PROFILING wrapper Error!")
            return models, optimizer, opt_param_scheduler
        print("OOTB_OPTIMIZER_PROFILING wrapper success!")
        recompute_parser = get_recompute_parser()
        recompute_parser.models = models
        optimizer.step = recompute_parser.hook_step_func(optimizer.step, models)

        if isinstance(models, list):
            for model in models:
                recompute_parser.construct_context_recursive("module", model, recompute_parser.context, True)
        else:
            recompute_parser.construct_context_recursive("module", models, recompute_parser.context, True)
        print("OOTB_OPTIMIZER-MODEL-PARSER: successfully hooking module")
        return models, optimizer, opt_param_scheduler

    return wrapper


def call_hook_func():
    print("success enter call_hook_func")
    recompute_parser = get_recompute_parser()
    models = recompute_parser.models
    if isinstance(models, list):
        for index, model in enumerate(models):
            recompute_parser.register_recursive_hook(model, recompute_parser.context,
                                                     recompute_parser.profiling_prefix, index)
    else:
        recompute_parser.register_recursive_hook(models, recompute_parser.context,
                                                 recompute_parser.profiling_prefix)


def allowed_recompute_parser_module_wrapper(allowed_recomputing_module):
    recomputing = get_recompute_parser()
    recomputing.add_allowed_recomputing_module(allowed_recomputing_module)


def cal_module_forward_time(context, event_dict: Dict[str, List]):
    cur_module_full_name = context.get('prefix_name', "") + '.' + context.get('name', "")
    if "memory" in context and cur_module_full_name in event_dict.keys():
        cur_module_event_list = event_dict.get(cur_module_full_name, [])
        for cur_level_event_list in cur_module_event_list:
            start_event = cur_level_event_list[0]
            end_event = cur_level_event_list[1]
            total_time = start_event.elapsed_time(end_event)

            context['forward_cnt'] = context.get('forward_cnt', 0) + 1
            context['pre_total_time'] = context.get('pre_total_time', 0) + total_time
            try:
                context['time'] = context['pre_total_time'] / context['forward_cnt']
            except ZeroDivisionError:
                context['time'] = 0

    if "layers" not in context:
        return
    for sub_layer_context in context["layers"]:
        cal_module_forward_time(sub_layer_context, event_dict)
