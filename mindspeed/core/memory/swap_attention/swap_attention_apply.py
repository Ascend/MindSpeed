# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from mindspeed.moe.utils import print_rank_0
from mindspeed.core.memory.swap_attention.prefetch import prefetch_register_post_backward_hook, prefetch_register_pre_forward_hook, get_swap_prefetch, get_layer_id
from mindspeed.core.memory.adaptive_recomputing.adaptive_recompute_apply import get_recompute_hook


class AdaptiveRecomputeSwapReg:
    """
        Register hook functions for each layer of the model according to the swap and recover strategy
    """

    def __init__(self):
        self.event_list2 = []

    def is_hook_layer(self, ctx, hook_list):
        return "name" in ctx and ctx["name"] in hook_list and "expert" not in ctx['prefix_name']

    def is_recompute_layer(self, ctx, prefetch_list):
        return "name" in ctx and "mlp" == ctx["name"] and get_layer_id(ctx["prefix_name"]) in prefetch_list

    def register_recursive_apply_prefetch(self, config, models, ctx, prefetch_recompute_group, prefetch_args):
        """
        Recursively register prefetch hooks and recompute strategies for model layers.

        Args:
            config (dict): Configuration dictionary containing layer context information
            models (nn.Module|list): Target model(s) to apply prefetch strategies
            ctx (dict): Execution context containing layer information
            prefetch_recompute_group (list): Triple containing [prefetch_layers, hook_layers, recompute_layers]
            prefetch_args (dict): Additional arguments for prefetch configuration
        """
        prefetch_list, hook_list, recompute_list = prefetch_recompute_group
        if not isinstance(prefetch_list[0], list):
            prefetch_layer = prefetch_list
            hook_layer = hook_list
            recompute_layer = recompute_list

        pre_layer_full_name = config["pre_layer_full_name"]
        pre_layer_ctx = config["pre_layer_ctx"]
        cur_layer_name = config["cur_layer_name"]
        swap_modules = config["swap_modules"]
        if cur_layer_name == "module" and isinstance(models, list):
            idx = 0
            for model in models:
                prefetch_layer = prefetch_list[idx] if isinstance(prefetch_list[0], list) else prefetch_list
                hook_layer = hook_list[idx] if isinstance(hook_list[0], list) else hook_list
                recompute_layer = recompute_list[idx] if isinstance(recompute_list[0], list) else recompute_list
                print_rank_0(f'prefetch_layer: {prefetch_layer}---{hook_layer}')
                if any(filter(None, prefetch_layer)):
                    prefetch_recompute_group = [prefetch_layer, hook_layer, recompute_layer]
                    self.register_recursive_apply_prefetch(config, model, self.get_list_layers_context(ctx, idx),
                                                    prefetch_recompute_group, prefetch_args)
                idx += 1
            return

        if self.is_hook_layer(ctx, hook_list):
            print_rank_0(f"prefetch forward and backward hook success: {pre_layer_full_name + '.' + cur_layer_name}")
            prefetch_register_post_backward_hook(models, pre_layer_full_name + '.' + cur_layer_name, prefetch_args)
            prefetch_register_pre_forward_hook(models, pre_layer_full_name + '.' + cur_layer_name, prefetch_args)
        if hook_list == prefetch_list and prefetch_list != ['']:
            if "name" in ctx and ctx["name"] in swap_modules and \
                    get_layer_id(ctx["prefix_name"]) in prefetch_list:
                print_rank_0(f"prefetch swap hook success: {pre_layer_full_name + '.' + cur_layer_name}")
                models.no_checkpoint_adaptive_recompute_forward = models.forward
                models.forward = get_swap_prefetch(prefetch_args).hook_swap_manager_forward(models.forward,
                                                                                            pre_layer_full_name +
                                                                                            '.' + cur_layer_name)
                get_recompute_hook().recompute_modules.append(models)
                return
            elif self.is_recompute_layer(ctx, recompute_list):
                print_rank_0(f"prefetch recompute hook success: {pre_layer_full_name + '.' + cur_layer_name}")
                models.no_checkpoint_adaptive_recompute_forward = models.forward
                models.forward = get_recompute_hook().hook_checkpoint_forward(models.forward)
                get_recompute_hook().recompute_modules.append(models)
                return
        else:
            if self.is_hook_layer(ctx, prefetch_list):
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
                "swap_modules": swap_modules,
            }
            prefetch_recompute_group = [prefetch_layer, hook_layer, recompute_layer]
            self.register_recursive_apply_prefetch(config, module, ctx['layers'][idx], prefetch_recompute_group, prefetch_args)
            idx += 1

    def get_list_layers_context(self, ctx, idx):
        current_ctx = {}
        for k, v in ctx.items():
            if k == "layers":
                current_ctx[k] = [v[idx]]
                continue
            current_ctx[k] = v
        return current_ctx