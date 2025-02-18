# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from collections.abc import Iterable
from copy import deepcopy

import re
import acl
import torch
import torch.nn

from megatron.training import print_rank_0, get_args
from megatron.core import tensor_parallel

from .adaptive_memory_tool import AdaptiveStepMgr, BYTES_PER_MB, SingletonBase, ContextKey as Key
from .adaptive_memory_tool import FuncLocationMgr
from .adaptive_memory_prefetch import AdaptiveMemoryPrefetch


class RecomputeHook(metaclass=SingletonBase):
    def __init__(self):
        self.recompute_modules = []

    @staticmethod
    def hook_checkpoint_forward(forward_func):
        def custom_forward(*args, **kwargs):
            def inside_forward(*new_args):
                origin_args = new_args[:len(args)]
                origin_kwargs = dict(zip(kwargs.keys(), new_args[len(args):]))
                return forward_func(*origin_args, **origin_kwargs)
            new_args = args + tuple(kwargs.values())
            return tensor_parallel.checkpoint(inside_forward, False, *new_args)
        return custom_forward

    def reset_recompute_modules(self):
        for m in self.recompute_modules:
            m.forward = m.no_checkpoint_adaptive_recompute_forward
        self.recompute_modules.clear()


class AdaptiveMemoryProfiling(metaclass=SingletonBase):

    def __init__(self):
        # saved module data and structure
        self.context = {'name': 'root', 'deep': 0, 'prefix_name': '', 'submodules': []}
        # save modules hook
        self.profiling_hooks = []
        # record allowed memory adaptation module
        self.allowed_adapt_module = []
        # time events, used to calculate time
        self.time_event_list = []
        # save origin modules
        self.checkpointed_modules = []
        self.layer0_module = None
        self.layer0_ctx = None

    def addup_allowed_mem_adapt_profiling_module(self, module):
        if not issubclass(module, torch.nn.Module):
            raise TypeError("Allowed adapt module must be subclass of torch.nn.Module")
        self.allowed_adapt_module.append(module)

    @staticmethod
    def _tag_module(ctx, current_ctx, current_is_adapt_module, upper_is_nn_module_list):
        if current_is_adapt_module:
            current_ctx[Key.ALLOWED_ADAPT] = True
            if upper_is_nn_module_list:
                ctx[Key.IS_MODULE_LIST] = True
                ctx[Key.IS_ADAPT_LAYER] = True
            else:
                current_ctx[Key.IS_ADAPT_LAYER] = True

            return False

        return True

    def record_time(self):
        while self.time_event_list:
            self._record_submodule_forward_time(self.context)

    def update_whole_model_memory(self):
        _, all_memory, _ = acl.rt.get_mem_info(1)
        self.context[Key.USED_MEM] = torch.npu.memory_allocated() / BYTES_PER_MB
        self.context[Key.DEVICE_MEMORY] = all_memory / BYTES_PER_MB

    def reset_profiling_all_hooks(self):
        self.reset_profiling_hooks()
        self.reset_profiling_recompute_hook()

    def reset_profiling_hooks(self):
        for ph in self.profiling_hooks:
            ph.remove()
        self.profiling_hooks.clear()

    def reset_profiling_recompute_hook(self):
        for m in self.checkpointed_modules:
            m.forward = m.no_checkpoint_forward
        self.checkpointed_modules.clear()

    def insert_func_profiling(self, ctx, child_name):
        self._find_adapt_layer(self.context, ctx, child_name)

    def _find_adapt_layer(self, ctx, new_ctx, child):
        if ctx.get(Key.ALLOWED_ADAPT, False):
            self._insert_ctx(ctx, new_ctx, child)
            return
        for sub in ctx.get(Key.SUBMODULES, []):
            self._find_adapt_layer(sub, new_ctx, child)

    @staticmethod
    def _is_parent_child_relation(parent_ctx, child_ctx):
        if parent_ctx[Key.DEEP] + 1 != child_ctx[Key.DEEP]:
            return False

        part1 = f"{parent_ctx[Key.PREFIX_NAME]}.{parent_ctx[Key.NAME]}".split(".")
        part2 = child_ctx[Key.PREFIX_NAME].split(".")
        if len(part1) != len(part2):
            return False

        # compare ctx parent cross chunks and layers, the prefix differ only with the index in torch.nn.ModuleList
        def compare(p1, p2):
            return re.sub(r'\d+$', '#', p1) == re.sub(r'\d+$', '#', p2)

        return all(compare(x, y) for x, y in zip(part1, part2))

    @staticmethod
    def _clone_to_insert_ctx(parent_ctx, new_ctx):
        cur_prefix_name = f"{parent_ctx[Key.PREFIX_NAME]}.{parent_ctx[Key.NAME]}"
        to_insert_ctx = deepcopy(new_ctx)
        if to_insert_ctx[Key.PREFIX_NAME] != cur_prefix_name:
            to_insert_ctx[Key.PREFIX_NAME] = cur_prefix_name
            del to_insert_ctx[Key.INPUT]
            del to_insert_ctx[Key.MEMORY]
            del to_insert_ctx[Key.PRE_TOTAL_TIME]
            del to_insert_ctx[Key.OUTPUT]
            del to_insert_ctx[Key.FORWARD_CNT]
            del to_insert_ctx[Key.AVG_TIME]
            del to_insert_ctx[Key.IS_MODLUE_OF_LAYER0]
        return to_insert_ctx

    def _insert_ctx(self, ctx, new_ctx, child_name):
        if self._is_parent_child_relation(ctx, new_ctx):
            to_insert_ctx = self._clone_to_insert_ctx(ctx, new_ctx)
            if child_name:
                idx = next(idx for idx, tmp in enumerate(ctx[Key.SUBMODULES]) if tmp[Key.NAME] == child_name)
                child_ctx = ctx[Key.SUBMODULES][idx]
                self._update_children_ctx(child_ctx, to_insert_ctx[Key.PREFIX_NAME], to_insert_ctx[Key.NAME])
                to_insert_ctx[Key.SUBMODULES] = [child_ctx]
                ctx[Key.SUBMODULES][idx] = to_insert_ctx
            else:
                siblings = ctx.get(Key.SUBMODULES, [])
                siblings.append(to_insert_ctx)
                ctx[Key.SUBMODULES] = siblings
            return True

        for sub in ctx.get(Key.SUBMODULES, []):
            if self._insert_ctx(sub, new_ctx, child_name):
                return True
        return False

    def _update_children_ctx(self, ctx, parent, func_name):
        old_prefix_name = ctx[Key.PREFIX_NAME]
        new_prefix_name = old_prefix_name[0:len(parent)] + "." + func_name + old_prefix_name[len(parent):]
        ctx[Key.PREFIX_NAME] = new_prefix_name
        ctx[Key.DEEP] += 1
        AdaptiveMemoryPrefetch().prefetch_deep_end = max(AdaptiveMemoryPrefetch().prefetch_deep_end, ctx[Key.DEEP])

        for sub in ctx.get(Key.SUBMODULES, []):
            self._update_children_ctx(sub, parent, func_name)

    def get_allowed_adapt_module(self):
        return self.allowed_adapt_module

    def is_layer0(self, ctx):
        if ctx[Key.NAME] == "0" and "expert" not in ctx[Key.PREFIX_NAME]:
            return True
        return False

    def forward_pre_hook(self, prefix, name, ctx):
        """ Hook, which will be registered before the FWD to add context parameters and add timer start event """
        def hook(module, *args, **kwargs):
            FuncLocationMgr().push_name(prefix, name)
            if Key.IS_LAYER0_OF_MODULE0 in ctx:
                FuncLocationMgr().is_first_layer = True

            if AdaptiveStepMgr().is_skipping_step():
                return

            if AdaptiveStepMgr().is_last_recompute_profiling_step():
                ctx[Key.INPUT] = self.cal_input_output_size(args) / BYTES_PER_MB
                mem_alloc = torch.npu.memory_allocated()
                ctx[Key.MEMORY] = mem_alloc / BYTES_PER_MB - ctx[Key.INPUT]
            else:
                # 通过Key.MEMORY来判断此module是否被执行
                ctx[Key.INPUT] = 0
                ctx[Key.MEMORY] = 0


            if AdaptiveStepMgr().is_recompute_profiling_step() and not AdaptiveStepMgr().is_last_recompute_profiling_step():
                start_event = torch.npu.Event(enable_timing=True)
                self.time_event_list.append([start_event])
                start_event.record()

        return hook

    def forward_post_hook(self, prefix, name, ctx):
        """ Hook, which will be registered in the FWD to calculate context parameters and add timer stop event """
        def hook(module, args, output):
            FuncLocationMgr().pop_name(prefix, name)
            if Key.IS_LAYER0_OF_MODULE0 in ctx:
                FuncLocationMgr().is_first_layer = False

            if AdaptiveStepMgr().is_recompute_profiling_step() and not AdaptiveStepMgr().is_last_recompute_profiling_step():
                end_event = torch.npu.Event(enable_timing=True)
                end_event.record()
                for item in reversed(self.time_event_list):
                    if len(item) == 1:
                        item.append(end_event)
                        break

            if AdaptiveStepMgr().is_last_recompute_profiling_step():
                ctx[Key.OUTPUT] = self.cal_input_output_size(output) / BYTES_PER_MB
                ctx[Key.MEMORY] = torch.npu.memory_allocated() / BYTES_PER_MB - ctx[Key.MEMORY]

        return hook

    def construct_ctx_recursively(self, deep, prefix_name, model, ctx, allowed_adapting):
        """ Function, recursively construct context to save profiling data in the future """
        next_allowed_adapting = allowed_adapting
        for name, module in model.named_children():
            if Key.SUBMODULES not in ctx:
                ctx[Key.SUBMODULES] = []
            current_ctx = {Key.NAME: name, Key.DEEP: deep, Key.PREFIX_NAME: prefix_name}
            ctx[Key.SUBMODULES].append(current_ctx)
            if self.is_layer0(current_ctx):
                AdaptiveMemoryPrefetch().prefetch_deep_start = current_ctx[Key.DEEP]
            if current_ctx[Key.DEEP] > AdaptiveMemoryPrefetch().prefetch_deep_end:
                AdaptiveMemoryPrefetch().prefetch_deep_end = current_ctx[Key.DEEP]
            if allowed_adapting:
                for allowed_adapt_module in self.allowed_adapt_module:
                    module_flag = isinstance(module, allowed_adapt_module)
                    model_flag = isinstance(model, torch.nn.ModuleList)
                    next_allowed_adapting = self._tag_module(ctx, current_ctx, module_flag, model_flag)
            next_name = (prefix_name + '.' + name) if prefix_name != '' else name
            next_deep = deep + 1
            self.construct_ctx_recursively(next_deep, next_name, module, current_ctx, next_allowed_adapting)

    def register_hook_recursively(self, model, ctx, in_first_module=False, in_first_layer=False, start_index=0):
        """ Function, recursively register hooks to get profiling data on needed modules """
        for module in model.children():
            if Key.SUBMODULES not in ctx:
                continue

            current_ctx = ctx[Key.SUBMODULES][start_index]
            name = current_ctx[Key.NAME]
            prefix_name = current_ctx[Key.PREFIX_NAME]

            # whole first module or in layer 0
            if prefix_name in (Key.MODULE, Key.MODULE + '0') or in_first_layer:
                if prefix_name not in (Key.MODULE, Key.MODULE + '0'):
                    current_ctx[Key.IS_MODLUE_OF_LAYER0] = True
                self._register_hook(module, prefix_name, name, current_ctx)
                self.register_hook_recursively(module, current_ctx, in_first_module, in_first_layer)
            # whole layer 0
            elif Key.ALLOWED_ADAPT in current_ctx and in_first_module and start_index == 0:
                self.layer0_ctx = current_ctx
                self.layer0_module = module
                current_ctx[Key.IS_LAYER0_OF_MODULE0] = True
                current_ctx[Key.IS_MODLUE_OF_LAYER0] = True
                self._register_hook(module, prefix_name, name, current_ctx)
                self.register_hook_recursively(module, current_ctx, in_first_module, True)
            # encoder
            elif isinstance(module, torch.nn.ModuleList) and Key.IS_ADAPT_LAYER in current_ctx and in_first_module:
                self._register_hook(model, ctx[Key.PREFIX_NAME], ctx[Key.NAME], ctx)
                self.register_hook_recursively(module, current_ctx, in_first_module, in_first_layer)
            # recompute layer hook
            elif Key.IS_MODULE_LIST in ctx and Key.ALLOWED_ADAPT in current_ctx:
                module.no_checkpoint_forward = module.forward
                module.forward = RecomputeHook().hook_checkpoint_forward(module.forward)
                self.checkpointed_modules.append(module)
            # do not hook, and check next one
            else:
                self.register_hook_recursively(module, current_ctx, in_first_module, in_first_layer)

            start_index += 1

    def cal_input_output_size(self, args):
        size = 0
        if isinstance(args, torch.Tensor):
            size += args.numel() * args.element_size()
        elif isinstance(args, Iterable):
            for arg in args:
                size += self.cal_input_output_size(arg)

        return size

    def _register_hook(self, module, prefix_name, name, current_ctx):
        pre_hook = module.register_forward_pre_hook(self.forward_pre_hook(prefix_name, name, current_ctx))
        post_hook = module.register_forward_hook(self.forward_post_hook(prefix_name, name, current_ctx))
        self.profiling_hooks.append(pre_hook)
        self.profiling_hooks.append(post_hook)

    def _record_submodule_forward_time(self, context):
        if Key.MEMORY in context and Key.IS_FUNCTION not in context:
            cur_event_list = self.time_event_list.pop(0)
            start, end = cur_event_list[0], cur_event_list[1]
            cur_time = start.elapsed_time(end)
            if Key.PRE_TOTAL_TIME in context:
                context[Key.FORWARD_CNT] += 1
                context[Key.PRE_TOTAL_TIME] += cur_time
                context[Key.AVG_TIME] = context[Key.PRE_TOTAL_TIME] / context[Key.FORWARD_CNT]
            else:
                context[Key.FORWARD_CNT] = 1
                context[Key.PRE_TOTAL_TIME] = cur_time
                context[Key.AVG_TIME] = cur_time
        if Key.SUBMODULES not in context:
            return
        for sub_layer in context[Key.SUBMODULES]:
            self._record_submodule_forward_time(sub_layer)

    def construct_and_register_profiling_hooks(self, models):
        """ Function, used to construct and register hooks into first model to get profiling data in the future """
        if isinstance(models, Iterable):
            for idx, model in enumerate(models):
                self.construct_ctx_recursively(1, Key.MODULE + str(idx), model, self.context, True)
                self.register_hook_recursively(model, self.context, idx == 0, start_index=idx)
        else:
            self.construct_ctx_recursively(1, Key.MODULE, models, self.context, True)
            self.register_hook_recursively(models, self.context)
