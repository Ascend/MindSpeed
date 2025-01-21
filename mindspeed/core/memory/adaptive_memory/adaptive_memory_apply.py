# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from typing import List
from copy import deepcopy
from megatron.training import print_rank_0
from .adaptive_memory_tool import SingletonBase, LayerAction, ModuleAction, ContextKey as Key
from .adaptive_memory_solver import AdaptMemGraphSolver
from .adaptive_memory_swap_manager import SwapManager, transformer_layer_register_post_forward_hook, transformer_layer_register_pre_backward_hook
from .adaptive_memory_profiling import RecomputeHook, AdaptiveMemoryProfiling


class AdaptMemApplyManager(metaclass=SingletonBase):

    def __init__(self):
        self.no_adapt_modules = []  # modules which don't join policy selections
        self.cur_module_index = 0  # DFS index

    # optype 0 save_to_cache, 1 apply_to_context
    def apply_op_to_context(self, adapt_policy_list: list, origin_context: dict):
        if len(adapt_policy_list) == 0:
            print_rank_0("adapt_policy_list Empty!")
            return origin_context
        context = deepcopy(origin_context)
        # 1.get all layers by order
        ordered_layers = []
        self.get_ordered_layers(context, ordered_layers, True)
        # 2.handle policy list
        idx = 0
        self.get_ordered_modules(ordered_layers[0][Key.SUBMODULES], [], 0)
        for policy in adapt_policy_list:
            n = policy[0]
            adapt_nodes = []
            if policy[1] == LayerAction.FULL_RECOMPUTE:
                status = ModuleAction.RECOMPUTE
                adapt_nodes = [status for _ in range(len(policy[2:]))]
                for i in range(idx, idx + n):
                    ordered_layers[i][ModuleAction.RECOMPUTE.name] = True
            elif policy[1] == LayerAction.FULL_SWAP:
                status = ModuleAction.SWAP
                adapt_nodes = [status for _ in range(len(policy[2:]))]
            elif policy[1] == LayerAction.ADAPTIVE:
                adapt_nodes = policy[2:]
            for i in range(idx, idx + n):
                self.apply_op_to_layer(ordered_layers[i], adapt_nodes, i)
            idx += n

        return context

    def apply_op_to_layer(self, ordered_layer, adapt_nodes: list, layer_index: int):
        if len(adapt_nodes) == 0:
            # don't need any operations if adapt_nodes is empty
            return
        # get all modules of the current layer through DFS
        ordered_module: List[dict] = []
        if Key.SUBMODULES not in ordered_layer:
            return
        self.cur_module_index = 0
        if layer_index == 0:
            self.no_adapt_modules.clear()
        self.get_ordered_modules(ordered_layer[Key.SUBMODULES], ordered_module, layer_index)

        for i, nodes in enumerate(adapt_nodes):
            if i >= len(ordered_module):
                break
            if Key.IS_FUNCTION in ordered_module[i]:
                func_action = nodes
                # add location infos for autofrad.function
                AdaptMemGraphSolver().add_func_locations(layer_index, ordered_module[i][Key.NAME], func_action)
                continue
            if nodes == ModuleAction.RECOMPUTE:
                ordered_module[i][ModuleAction.RECOMPUTE.name] = True
            elif nodes == ModuleAction.SWAP:
                ordered_module[i][ModuleAction.SWAP.name] = True

    def get_ordered_layers(self, model: dict, ordered_layers: list, is_root_layer: bool = False):
        # root module may have multiple layers due to vpp parallel
        if is_root_layer:
            if Key.SUBMODULES not in model:
                return
            for sub_model in model[Key.SUBMODULES]:
                self.get_ordered_layers(sub_model, ordered_layers)
            return

        if Key.IS_ADAPT_LAYER in model:
            for sub_layer in model[Key.SUBMODULES]:
                ordered_layers.append(sub_layer)
        if Key.SUBMODULES not in model:
            return
        for sub_model in model[Key.SUBMODULES]:
            self.get_ordered_layers(sub_model, ordered_layers)

    def get_ordered_modules(self, layer: dict, ordered_modules: list, layer_index: int):
        for sub_layer in layer:
            # The first layer judges through ['memory']
            if layer_index == 0:
                if Key.MEMORY in sub_layer:
                    ordered_modules.append(sub_layer)
                else:
                    # use the DFS index as the unique identifier
                    self.no_adapt_modules.append(self.cur_module_index)
            else:
                if self.cur_module_index not in self.no_adapt_modules:
                    ordered_modules.append(sub_layer)

            self.cur_module_index += 1
            if Key.SUBMODULES in sub_layer:
                self.get_ordered_modules(sub_layer[Key.SUBMODULES], ordered_modules, layer_index)

    def apply_hook_to_model(self, models, context, pre_context, is_root_layer: bool = False):
        if is_root_layer and isinstance(models, list):
            layer_idx = 0
            for model in models:
                self.apply_hook_to_model(model, get_cur_layer_context(context, layer_idx), context)
                layer_idx += 1
            return
        # pass autograd.function
        if Key.IS_FUNCTION in context:
            if Key.SUBMODULES in context:
                for i in range(0, len(context[Key.SUBMODULES])):
                    self.apply_hook_to_model(models, context[Key.SUBMODULES][i], context)
            return
        # apply hooks for recompute models
        if context.get(ModuleAction.RECOMPUTE.name, False):
            models.no_checkpoint_adaptive_recompute_forward = models.forward
            models.forward = RecomputeHook().hook_checkpoint_forward(models.forward)
            RecomputeHook().recompute_modules.append(models)
            print_rank_0('recompute hooked on %s' % models._get_name())
            return
        # apply hooks for swap modules
        if context.get(ModuleAction.SWAP.name, False):
            SwapManager().hook_prefetch_forward(models, '')
            print_rank_0('swap hooked on %s' % models._get_name())
            return
        # apply hooks for oom swap
        if Key.ALLOWED_ADAPT in context:
            transformer_layer_register_post_forward_hook(models)
            transformer_layer_register_pre_backward_hook(models)
            SwapManager().hook_oom_rescue_forward(models)
            print_rank_0('oom rescue hooked on %s' % models._get_name())

        module_idx = 0
        for name, module in models.named_children():
            self.apply_hook_to_model(module, context[Key.SUBMODULES][module_idx], context)
            module_idx += 1

    def apply_new_adapt_policy(self, adapt_policy_list, context, models):
        AdaptMemGraphSolver().func_locations.clear()
        new_context = self.apply_op_to_context(adapt_policy_list, context)
        self.apply_hook_to_model(models, new_context, "", True)


# get layer by idx in root module
def get_cur_layer_context(context, idx):
    current_context = {}
    for k, v in context.items():
        if k == Key.SUBMODULES:
            current_context[k] = [v[idx]]
            continue
        current_context[k] = v
    return current_context
