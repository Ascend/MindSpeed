# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import re
import torch
from megatron.training import print_rank_0, get_args
from .adaptive_memory_tool import SingletonBase, FuncLocationMgr, broadcast_obj
from .adaptive_memory_tool import AdaptiveStepMgr, ContextKey as Key
from .adaptive_memory_swap_manager import SwapManager, transformer_layer_register_post_forward_hook, \
    transformer_layer_register_pre_backward_hook, LayerProfilingHook


class AdaptiveMemoryPrefetch(metaclass=SingletonBase):
    def __init__(self):
        self.modules_hooks = []
        self.is_stable_apply = False
        self.is_first_select_module = False
        self.config = {
            "pre_layer_full_name": "",
            "cur_layer_name": "module",
        }
        self.chunk_num = 0
        self.forward_time = 0
        self.swap_time = 0
        self.not_need_swap_module = []
        self.need_swap_module_full_name = []
        self.need_swap_module_name = []
        self.need_swap_module_ctx = []
        self.prefetch_module_dict = {}
        self.abnormal_scenario_module_list = ["input_norm", "self_attention", "post_attention_norm"]
        # 统计数据
        self.prefetch_hook_interval = None
        self.prefetch_deep_list = []
        self.prefetch_deep_start = 0
        self.prefetch_deep_end = 0
        self.each_depth_run_times = 2
        self.layer_list = []
        self.swap_event_dict = {}
        self.swap_memory_in_module_dict = {}
        self.prefetch_module_event_dict = {}
        # auto_function
        self.function_swap_profiling_deep = 0
        self.function_list = []
        self.prefetch_function_list = []

    def reset_prefetch_hooks(self):
        SwapManager().reset_prefetch_hooked_modules()

    def reset_module_hooks(self):
        for hook_handle in self.modules_hooks:
            hook_handle.remove()
        self.modules_hooks.clear()

    def reset_adaptive_prefetch_all_hooks(self):
        self.reset_prefetch_hooks()
        self.reset_module_hooks()
        SwapManager().reset_post_layer_forward_and_pre_layer_backward_hooks()
        LayerProfilingHook().reset_layer_profiling_hook()

    def set_forward_time(self):
        self.forward_time = SwapManager().forward_time

    def _get_list_layers_context(self, ctx, idx):
        current_ctx = {}
        for k, v in ctx.items():
            if k == Key.SUBMODULES:
                current_ctx[k] = [v[idx]]
                continue
            current_ctx[k] = v
        return current_ctx

    def is_parent_module(self, key, keys):
        if self.need_swap_module_name[key][-1] not in keys:
            return True
        else:
            if not self.need_swap_module_name[self.need_swap_module_name[key][-1]][0]:
                return True
            else:
                return False

    # get prefetch config
    def solve_prefetch_config(self):
        self.prefetch_deep_list = [num for num in range(self.prefetch_deep_start, self.prefetch_deep_end + 1) for _ in range(self.each_depth_run_times)]
        self.prefetch_hook_interval = len(self.prefetch_deep_list)
        self.set_chunk_num()

    def set_chunk_num(self):
        all_args = get_args()
        pp_size = all_args.pipeline_model_parallel_size or 1
        vpp_size = all_args.virtual_pipeline_model_parallel_size or 1
        num_prefetch = all_args.num_layers // pp_size
        self.layer_list = [str(num) for num in range(0, num_prefetch)]
        if vpp_size > 1:
            if vpp_size <= num_prefetch:
                self.chunk_num = vpp_size
            else:
                self.chunk_num = num_prefetch
        else:
            self.chunk_num = 1

    def get_deep_index(self):
        step = AdaptiveStepMgr().skip_steps + AdaptiveStepMgr().recompute_profiling_steps
        return (AdaptiveStepMgr().get_cur_step() - step) % self.prefetch_hook_interval

    # profiling for layer0
    def prefetch_profiling_register(self, ctx, models, cur_layer_full_name):
        if self.prefetch_deep_list[self.get_deep_index()] == ctx[Key.DEEP] and ctx.get(Key.IS_MODLUE_OF_LAYER0, False):
            prefetch_register_forward_hook_for_recording_time(models, cur_layer_full_name)
            prefetch_register_pre_forward_hook(models, cur_layer_full_name)
            # register pack/unpack
            print_rank_0(f"cur_step()={AdaptiveStepMgr().get_cur_step()}, is_recording=True, prefetch swap hook success: {cur_layer_full_name}")
            SwapManager().hook_prefetch_forward(models, cur_layer_full_name)

        if ctx.get(Key.IS_LAYER0_OF_MODULE0, False):
            print_rank_0(f"cur_step()={AdaptiveStepMgr().get_cur_step()}, is_recording=True, prefetch forward and backward hook success: {cur_layer_full_name}")
            prefetch_register_pre_forward_hook(models, cur_layer_full_name, True)
            transformer_layer_register_post_forward_hook(models, True)
            transformer_layer_register_pre_backward_hook(models)

    def prefetch_profiling_register_for_function(self, ctx, cur_layer_full_name):
        if self.prefetch_deep_list[self.get_deep_index()] == ctx[Key.DEEP]:
            self.function_swap_profiling_deep = ctx[Key.DEEP]
            print_rank_0(f"cur_step()={AdaptiveStepMgr().get_cur_step()}, {self.function_swap_profiling_deep=}, is_recording=True, prefetch swap hook success: {cur_layer_full_name}")

    def prefetch_register(self, ctx, models, cur_layer_full_name):
        if ctx.get(Key.IS_LAYER0_OF_MODULE0, False):
            print_rank_0(f"is_recording=False, prefetch forward and backward hook success: cur_step()={AdaptiveStepMgr().get_cur_step()}, {cur_layer_full_name}")
            transformer_layer_register_post_forward_hook(models)
            transformer_layer_register_pre_backward_hook(models)
            from .adaptive_memory_profiling import AdaptiveMemoryProfiling
            LayerProfilingHook().apply_layer_profiling_hook(models)
        if cur_layer_full_name in self.need_swap_module_name:
            print_rank_0(f"is_recording=False, prefetch swap hook success: cur_step()={AdaptiveStepMgr().get_cur_step()}, {cur_layer_full_name}")
            SwapManager().hook_prefetch_forward(models, cur_layer_full_name)
            ctx[Key.IS_SWAP] = True
        elif Key.AVG_TIME in ctx and Key.IS_MODLUE_OF_LAYER0 in ctx:
            ctx[Key.IS_SWAP] = False

    def prefetch_register_for_function(self, ctx, cur_layer_full_name):
        if cur_layer_full_name in self.need_swap_module_name:
            if ctx[Key.NAME] not in self.prefetch_function_list:
                print_rank_0(f"is_recording=False, prefetch swap hook success: cur_step()={AdaptiveStepMgr().get_cur_step()}, {cur_layer_full_name}")
                self.prefetch_function_list.append(ctx[Key.NAME])
            ctx[Key.IS_SWAP] = True
        else:
            ctx[Key.IS_SWAP] = False


    def register_recursive_apply_prefetch(self, config, models, ctx, is_prefetch_prof=True):
        pre_layer_full_name = config["pre_layer_full_name"]
        cur_layer_name = config["cur_layer_name"]
        if cur_layer_name == Key.MODULE and isinstance(models, list):
            idx = 0
            for model in models:
                if idx < self.chunk_num:
                    self.register_recursive_apply_prefetch(config, model, self._get_list_layers_context(ctx, idx), is_prefetch_prof)
                idx += 1
            return

        # deal auto_function
        if ctx.get(Key.IS_FUNCTION, False):
            cur_layer_full_name = pre_layer_full_name + "." + ctx[Key.NAME]
            if is_prefetch_prof:
                # function profiling
                self.prefetch_profiling_register_for_function(ctx, cur_layer_full_name)
            else:
                # function prefetch
                self.prefetch_register_for_function(ctx, cur_layer_full_name)

            config = {
                "pre_layer_full_name": cur_layer_full_name,
                "cur_layer_name": cur_layer_name,
            }
            self.register_recursive_apply_prefetch(config, models, ctx[Key.SUBMODULES][0], is_prefetch_prof)
            return
        cur_layer_full_name = pre_layer_full_name + '.' + cur_layer_name

        if is_prefetch_prof:
            self.prefetch_profiling_register(ctx, models, cur_layer_full_name)
        else:
            self.prefetch_register(ctx, models, cur_layer_full_name)

        pre_layer_full_name = ctx[Key.PREFIX_NAME] + "." + ctx[Key.NAME]
        idx = 0
        for name, module in models.named_children():
            config = {
                "pre_layer_full_name": pre_layer_full_name,
                "cur_layer_name": name,
            }
            self.register_recursive_apply_prefetch(config, module, ctx[Key.SUBMODULES][idx], is_prefetch_prof)
            idx += 1

    def _get_swappable_child_ctx(self, module_ctx):
        res_ctxs, res_names = [], []
        for child_ctx in module_ctx.get(Key.SUBMODULES, []):
            if Key.AVG_TIME in child_ctx:
                res_ctxs.append(child_ctx)
                res_names.append(child_ctx[Key.PREFIX_NAME] + '.' + child_ctx[Key.NAME])
            else:
                sub_res_ctxs, sub_res_names = self._get_swappable_child_ctx(child_ctx)
                res_ctxs.extend(sub_res_ctxs)
                res_names.extend(sub_res_names)
        return res_ctxs, res_names

    def adjust_need_swap_module(self):
        if len(self.need_swap_module_name) > 0:
            last_module_ctx = self.need_swap_module_ctx.pop()
            self.need_swap_module_name.pop()
            child_module_ctxs, child_module_names = self._get_swappable_child_ctx(last_module_ctx)
            self.need_swap_module_ctx.extend(child_module_ctxs)
            self.need_swap_module_name.extend(child_module_names)

    def is_no_module_to_swap(self):
        return len(self.need_swap_module_name) == 0

    def record_prefetch_time(self, context):
        if len(list(self.prefetch_module_event_dict.keys())) == 0:
            return
        first_key = list(self.prefetch_module_event_dict.keys())[0]
        if Key.PREFIX_NAME in context and Key.NAME in context and first_key == context[Key.PREFIX_NAME] + "." + context[Key.NAME]:
            cur_event_list = self.prefetch_module_event_dict.pop(first_key)
            for event_list in cur_event_list:
                start, end = event_list[0], event_list[1]
                cur_time = start.elapsed_time(end)
                if Key.MODULE_FORWARD_TOTAL_TIME in context:
                    context[Key.MODULE_FORWARD_CNT] += 1
                    context[Key.MODULE_FORWARD_TOTAL_TIME] += cur_time
                    context[Key.MODULE_FORWARD_AVG_TIME] = context[Key.MODULE_FORWARD_TOTAL_TIME] / context[Key.MODULE_FORWARD_CNT]
                else:
                    context[Key.MODULE_FORWARD_CNT] = 1
                    context[Key.MODULE_FORWARD_TOTAL_TIME] = cur_time
                    context[Key.MODULE_FORWARD_AVG_TIME] = cur_time
        if Key.SUBMODULES not in context:
            return
        for submodule in context[Key.SUBMODULES]:
            self.record_prefetch_time(submodule)

    def record_swap_time(self, context):
        if len(list(self.swap_event_dict.keys())) == 0:
            return
        first_key = list(self.swap_event_dict.keys())[0]
        if Key.PREFIX_NAME in context and Key.NAME in context and first_key == context[Key.PREFIX_NAME] + "." + context[Key.NAME]:
            cur_event_list = self.swap_event_dict.pop(first_key)
            for event_list in cur_event_list:
                start, end = event_list[0], event_list[1]
                cur_time = start.elapsed_time(end)
                if Key.MODULE_SWAP_TOTAL_TIME in context:
                    context[Key.MODULE_SWAP_CNT] += 1
                    context[Key.MODULE_SWAP_TOTAL_TIME] += cur_time
                    context[Key.MODULE_SWAP_AVG_TIME] = context[Key.MODULE_SWAP_TOTAL_TIME] / context[Key.MODULE_SWAP_CNT]
                else:
                    context[Key.MODULE_SWAP_CNT] = 1
                    context[Key.MODULE_SWAP_TOTAL_TIME] = cur_time
                    context[Key.MODULE_SWAP_AVG_TIME] = cur_time
        if Key.SUBMODULES not in context:
            return
        for submodule in context[Key.SUBMODULES]:
            self.record_swap_time(submodule)

    def record_swap_memory(self, context):
        if len(list(self.swap_memory_in_module_dict.keys())) == 0:
            return
        first_key = list(self.swap_memory_in_module_dict.keys())[0]
        if Key.PREFIX_NAME in context and Key.NAME in context and first_key == context[Key.PREFIX_NAME] + "." + context[Key.NAME]:
            memory = self.swap_memory_in_module_dict.pop(first_key)
            if Key.MODULE_SWAP_TOTAL_MEMORY in context:
                context[Key.MODULE_SWAP_TOTAL_MEMORY] += memory
                context[Key.MODULE_SWAP_AVG_MEMORY] = context[Key.MODULE_SWAP_TOTAL_MEMORY] / context[Key.MODULE_SWAP_CNT]
            else:
                context[Key.MODULE_SWAP_TOTAL_MEMORY] = memory
                context[Key.MODULE_SWAP_AVG_MEMORY] = context[Key.MODULE_SWAP_TOTAL_MEMORY] / context[Key.MODULE_SWAP_CNT]
        if Key.SUBMODULES not in context:
            return
        for submodule in context[Key.SUBMODULES]:
            self.record_swap_memory(submodule)

    def deal_not_need_swap_module(self, context):

        if context.get(Key.IS_MODLUE_OF_LAYER0, False) and Key.IS_SWAP not in context:
            context[Key.IS_SWAP] = False

        if Key.IS_SWAP in context and not context[Key.IS_SWAP]:
            self.not_need_swap_module.append(context[Key.PREFIX_NAME] + "." + context[Key.NAME])

        if Key.SUBMODULES not in context:
            return

        for submodule in context[Key.SUBMODULES]:
            self.deal_not_need_swap_module(submodule)

    def clear_dict(self):
        self.prefetch_module_event_dict.clear()
        self.swap_event_dict.clear()
        self.swap_memory_in_module_dict.clear()

    def update_ctx(self, models, context):
        if self.get_deep_index() % self.each_depth_run_times == 0:
            self.record_prefetch_time(context)
            self.record_swap_time(context)
            self.record_swap_memory(context)
        # 清除所有钩子
        self.reset_adaptive_prefetch_all_hooks()
        # 重新挂hook
        if not AdaptiveStepMgr().is_swap_profiling_done():
            self.register_recursive_apply_prefetch(self.config, models, context)
        # 清空dict
        self.clear_dict()

    def init_swap_modules(self, context):
        if Key.IS_LAYER0_OF_MODULE0 in context:
            for child_ctx in context[Key.SUBMODULES]:
                if Key.AVG_TIME in child_ctx:
                    self.need_swap_module_name.append(child_ctx[Key.PREFIX_NAME] + '.' + child_ctx[Key.NAME])
                    self.need_swap_module_ctx.append(child_ctx)
            return
        for child_ctx in context.get(Key.SUBMODULES, []):
            self.init_swap_modules(child_ctx)

    def adaptive_select_module(self, models, context):
        if len(self.need_swap_module_name) == 0:
            # 估计需要swap的module
            self.set_forward_time()
            self.init_swap_modules(context)
            self.need_swap_module_name = broadcast_obj(self.need_swap_module_name)

        if self.is_first_select_module and SwapManager().is_need_adjust_module():
            # 微调swap module
            print_rank_0(f"start adjust swap module, forward time is {LayerProfilingHook().get_single_layer_time()}")
            self.adjust_need_swap_module()
            if self.is_no_module_to_swap():
                # 处理异常场景
                self.is_stable_apply = True
        elif self.is_first_select_module and not SwapManager().is_need_adjust_module():
            print_rank_0(f"swap is stable, step={AdaptiveStepMgr().get_cur_step()}, "
                         f"forward time is {LayerProfilingHook().get_single_layer_time()}")
            self.is_stable_apply = True

        self.is_first_select_module = True
        # 移除preftech的所有hook
        self.reset_adaptive_prefetch_all_hooks()
        # 重新挂preftech的钩子
        self.register_recursive_apply_prefetch(self.config, models, context, False)
        # 清空
        self.clear_dict()
        LayerProfilingHook().forward_time_list.clear()

    def sync_d2h_for_recording_time(self, module_name, is_function=False):
        # 每个module前向结束后插入end_event
        module_forward_end_event = torch.npu.Event(enable_timing=True)
        module_forward_end_event.record()
        self.prefetch_module_event_dict[module_name][-1].append(module_forward_end_event)

        torch.cuda.current_stream().wait_stream(SwapManager().prefetch_stream)
        end_pack_event = None
        if AdaptiveStepMgr().is_swap_profiling_step():
            end_pack_event = torch.npu.Event(enable_timing=True)
            end_pack_event.record()

        for swap_tensor in SwapManager().swap_tensor_in_module:
            # 更新每个tensor的pack_module_name
            swap_tensor.pack_module_name = SwapManager().swap_tensors[-1].layer_name
            if swap_tensor is SwapManager().swap_tensor_in_module[0]:
                swap_tensor.first_tensor = True
            swap_tensor.end_pack_event = end_pack_event

        # record swap info
        for swap_tensor in SwapManager().swap_tensor_in_module:
            # cal tensor memory (MB)
            tensor_memory = (swap_tensor.tensor.numel() * swap_tensor.tensor.element_size()) / (1024 * 1024)

            if swap_tensor.pack_module_name == module_name:
                self.recording_swap_momery_in_module(swap_tensor, swap_tensor.pack_module_name, tensor_memory)
                self.recording_swap_time_in_module(swap_tensor, swap_tensor.pack_module_name, is_function)
            else:
                self.recording_swap_momery_in_module(swap_tensor, module_name, tensor_memory)
                self.recording_swap_time_in_module(swap_tensor, module_name, is_function)

        # reset swap_tensor_in_module
        SwapManager().swap_tensor_in_module = []

    def is_module_in_need_swap_module_name(self, module_name):
        if module_name in self.need_swap_module_name:
            return module_name
        return None

    # Records the memory swapped to the cpu in module
    def recording_swap_momery_in_module(self, swap_tensor, key, tensor_memory):
        has_key = key in AdaptiveMemoryPrefetch().swap_memory_in_module_dict.keys()
        if not has_key:
            AdaptiveMemoryPrefetch().swap_memory_in_module_dict[key] = tensor_memory
        else:
            if not swap_tensor.is_slice_tensor:
                AdaptiveMemoryPrefetch().swap_memory_in_module_dict[key] += tensor_memory

    # Records the time swapped to the cpu in module
    def recording_swap_time_in_module(self, swap_tensor, key, is_function):
        has_key = key in AdaptiveMemoryPrefetch().swap_event_dict.keys()
        if not has_key and swap_tensor.first_tensor:
            AdaptiveMemoryPrefetch().swap_event_dict[key] = [[swap_tensor.start_pack_event, swap_tensor.end_pack_event]]
        elif has_key and swap_tensor.first_tensor:
            if is_function:
                AdaptiveMemoryPrefetch().swap_event_dict[key].append([swap_tensor.start_pack_event, swap_tensor.end_pack_event])
            else:
                AdaptiveMemoryPrefetch().swap_event_dict[swap_tensor.pack_module_name].append([swap_tensor.start_pack_event, swap_tensor.end_pack_event])


def forward_post_hook_func_for_recording_time(module_name):
    def custom_func(module, *args, **kargs):
        AdaptiveMemoryPrefetch().sync_d2h_for_recording_time(module_name)

    return custom_func


def pre_forward_func(module_name, is_mark_first_layer):
    if is_mark_first_layer:
        FuncLocationMgr().is_first_layer = True
    else:
        module_forward_start_event = torch.npu.Event(enable_timing=True)
        if module_name not in AdaptiveMemoryPrefetch().prefetch_module_event_dict.keys():
            AdaptiveMemoryPrefetch().prefetch_module_event_dict[module_name] = [[module_forward_start_event]]
        else:
            AdaptiveMemoryPrefetch().prefetch_module_event_dict[module_name].append([module_forward_start_event])
        module_forward_start_event.record()


def pre_forward_hook_func(module_name, is_mark_first_layer):
    def custom_func(module, *args, **kargs):
        pre_forward_func(module_name, is_mark_first_layer)

    return custom_func


def prefetch_register_forward_hook_for_recording_time(module, name):
    post_hook = module.register_forward_hook(forward_post_hook_func_for_recording_time(name))
    AdaptiveMemoryPrefetch().modules_hooks.append(post_hook)


def prefetch_register_pre_forward_hook(module, name, is_mark_first_layer=False):
    pre_hook = module.register_forward_pre_hook(pre_forward_hook_func(name, is_mark_first_layer))
    AdaptiveMemoryPrefetch().modules_hooks.append(pre_hook)