# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import torch
from megatron.core import tensor_parallel
from megatron.training import print_rank_0
from megatron.training import get_args
from mindspeed.core.memory.adaptive_recomputing.swap_manager import SwapManager
from mindspeed.core.memory.adaptive_recomputing.prefetch import prefetch_tensor, prefetch_register_post_backward_hook, prefetch_register_pre_forward_hook, get_swap_prefetch, get_layer_id


class RecomputeHook:
    recompute_hook = None

    def __init__(self):
        self.recompute_modules = []

    @staticmethod
    def hook_checkpoint_forward(forward_func):
        def custom_forward(*args, **kargs):
            def inside_forward(*args):
                return forward_func(*args, **kargs)

            return tensor_parallel.checkpoint(inside_forward, None, *args)

        return custom_forward

    def reset_recompute_modules(self):
        for m in self.recompute_modules:
            m.forward = m.no_checkpoint_adaptive_recompute_forward
        self.recompute_modules.clear()


def get_recompute_hook():
    if RecomputeHook.recompute_hook is None:
        RecomputeHook.recompute_hook = RecomputeHook()
    return RecomputeHook.recompute_hook


class SwapManagerHook:
    swap_hook = None

    def __init__(self):
        self.tensor_layer_name_prefix = ""
        self.pre_tensor_layer_name_prefix = ""
        self.swap_manager_modules = []

    @staticmethod
    def unpack_hook(data):
        return SwapManager().unwrap_tensor(data)

    def pack_hook(self, origin_tensor):
        pre_tensor_is_allowed_swap = False
        # enter diff layer, make other layer tensor status to can be swapped
        if self.tensor_layer_name_prefix != self.pre_tensor_layer_name_prefix:
            pre_tensor_is_allowed_swap = True
            self.pre_tensor_layer_name_prefix = self.tensor_layer_name_prefix
        return SwapManager().wrap_tensor(origin_tensor, pre_tensor_is_allowed_swap)

    def hook_swap_manager_forward(self, forward_func, layer_name_prefix):
        def custom_forward(*args, **kargs):
            self.tensor_layer_name_prefix = layer_name_prefix
            with torch.autograd.graph.saved_tensors_hooks(self.pack_hook, self.unpack_hook):
                return forward_func(*args, **kargs)

        return custom_forward

    def reset_tensor_layer_info(self):
        self.tensor_layer_name_prefix = ""
        self.pre_tensor_layer_name_prefix = ""

    def reset_swap_manager_modules(self):
        for m in self.swap_manager_modules:
            m.forward = m.no_checkpoint_swap_forward
        self.swap_manager_modules.clear()


def get_swap_hook():
    if SwapManagerHook.swap_hook is None:
        SwapManagerHook.swap_hook = SwapManagerHook()
    return SwapManagerHook.swap_hook


def register_recursive_apply(config, models, ctx):
    pre_layer_ctx = config["pre_layer_ctx"]
    cur_layer_name = config["cur_layer_name"]
    if cur_layer_name == "module" and isinstance(models, list):
        idx = 0
        for model in models:
            register_recursive_apply(config, model, get_list_layers_context(ctx, idx))
            idx += 1
        return

    if 'recompute' in ctx and ctx['recompute']:
        models.no_checkpoint_adaptive_recompute_forward = models.forward
        models.forward = get_recompute_hook().hook_checkpoint_forward(models.forward)
        get_recompute_hook().recompute_modules.append(models)
        return

    if 'allowed_recomputing' in pre_layer_ctx:
        models.no_checkpoint_swap_forward = models.forward
        models.forward = get_swap_hook().hook_swap_manager_forward(models.forward, ctx["prefix_name"])
        get_swap_hook().swap_manager_modules.append(models)
        return

    idx = 0
    for name, module in models.named_children():
        config = {
            "pre_layer_ctx": ctx,
            "cur_layer_name": name,
        }
        register_recursive_apply(config, module, ctx['layers'][idx])
        idx += 1


def is_hook_layer(ctx, hook_list):
    if "name" in ctx and ctx["name"] in hook_list and "expert" not in ctx['prefix_name']:
        return True
    return False


def is_recompute_layer(ctx, prefetch_list):
    if "name" in ctx and "mlp" == ctx["name"] and get_layer_id(ctx["prefix_name"]) in prefetch_list:
        return True
    return False


def register_recursive_apply_prefetch(config, models, ctx, prefetch_recompute_group, prefetch_args):
    args = get_args()
    prefetch_list, hook_list, recompute_list = prefetch_recompute_group
    if not isinstance(prefetch_list[0], list):
        prefetch_layer = prefetch_list
        hook_layer = hook_list
        recompute_layer = recompute_list

    pre_layer_full_name = config["pre_layer_full_name"]
    pre_layer_ctx = config["pre_layer_ctx"]
    cur_layer_name = config["cur_layer_name"]
    if cur_layer_name == "module" and isinstance(models, list):
        idx = 0
        for model in models:
            prefetch_layer = prefetch_list[idx] if isinstance(prefetch_list[0], list) else prefetch_list
            hook_layer = hook_list[idx] if isinstance(hook_list[0], list) else hook_list
            recompute_layer = recompute_list[idx] if isinstance(recompute_list[0], list) else recompute_list
            print_rank_0(f'prefetch_layer: {prefetch_layer}---{hook_layer}')
            if any(filter(None, prefetch_layer)):
                prefetch_recompute_group = [prefetch_layer, hook_layer, recompute_layer]
                register_recursive_apply_prefetch(config, model, get_list_layers_context(ctx, idx),
                                                  prefetch_recompute_group, prefetch_args)
            idx += 1
        return

    if is_hook_layer(ctx, hook_list):
        print_rank_0(f"prefetch forward and backward hook success: {pre_layer_full_name + '.' + cur_layer_name}")
        prefetch_register_post_backward_hook(models, pre_layer_full_name + '.' + cur_layer_name, prefetch_args)
        prefetch_register_pre_forward_hook(models, pre_layer_full_name + '.' + cur_layer_name, prefetch_args)
    if hook_list == prefetch_list and prefetch_list != ['']:
        if "name" in ctx and ctx["name"] in args.swap_modules and \
                get_layer_id(ctx["prefix_name"]) in prefetch_list:
            print_rank_0(f"prefetch swap hook success: {pre_layer_full_name + '.' + cur_layer_name}")
            models.no_checkpoint_adaptive_recompute_forward = models.forward
            models.forward = get_swap_prefetch(prefetch_args).hook_swap_manager_forward(models.forward,
                                                                                        pre_layer_full_name +
                                                                                        '.' + cur_layer_name)
            get_recompute_hook().recompute_modules.append(models)
            return
        elif is_recompute_layer(ctx, recompute_list):
            print_rank_0(f"prefetch recompute hook success: {pre_layer_full_name + '.' + cur_layer_name}")
            models.no_checkpoint_adaptive_recompute_forward = models.forward
            models.forward = get_recompute_hook().hook_checkpoint_forward(models.forward)
            get_recompute_hook().recompute_modules.append(models)
            return
    else:
        if is_hook_layer(ctx, prefetch_list):
            print_rank_0(f"prefetch tensor hook success: {pre_layer_full_name + '.' + cur_layer_name}")
            models.no_checkpoint_adaptive_recompute_forward = models.forward
            models.forward = get_swap_prefetch(prefetch_args).hook_swap_manager_forward(models.forward,
                                                                                        pre_layer_full_name +
                                                                                        '.' + cur_layer_name)
            get_recompute_hook().recompute_modules.append(models)
            return
    pre_layer_full_name += "." + cur_layer_name if pre_layer_full_name != "" else cur_layer_name
    idx = 0
    for name, module in models.named_children():
        config = {
            "pre_layer_full_name": pre_layer_full_name,
            "pre_layer_ctx": ctx,
            "cur_layer_name": name,
        }
        prefetch_recompute_group = [prefetch_layer, hook_layer, recompute_layer]
        register_recursive_apply_prefetch(config, module, ctx['layers'][idx], prefetch_recompute_group, prefetch_args)
        idx += 1


def get_list_layers_context(ctx, idx):
    current_ctx = {}
    for k, v in ctx.items():
        if k == "layers":
            current_ctx[k] = [v[idx]]
            continue
        current_ctx[k] = v
    return current_ctx