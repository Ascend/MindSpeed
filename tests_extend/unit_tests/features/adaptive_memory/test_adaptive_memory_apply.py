from mindspeed import megatron_adaptor
from mindspeed.core.memory.adaptive_memory.adaptive_memory_apply import AdaptMemApplyManager
from mindspeed.core.memory.adaptive_memory.adaptive_memory_profiling import RecomputeHook
from mindspeed.core.memory.adaptive_memory.adaptive_memory_tool import ModuleAction, LayerAction, ContextKey as Key
import torch.nn as nn

from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from megatron.core.num_microbatches_calculator import get_num_microbatches, init_num_microbatches_calculator
from megatron.core import parallel_state as ps


class SimpleModel(nn.Module):
    def __init__(self, inputt, hidden, output):
        super(SimpleModel, self).__init__()
        self.module = CustomModule1(inputt, hidden, output)

    def forward(self, X):
        output = self.module(X)
        return output


class CustomModule1(nn.Module):
    def __init__(self, inputt, hidden, output):
        super(CustomModule1, self).__init__()
        self.module = CustomModule2(inputt, hidden, output)

    def forward(self, X):
        output = self.module(X)
        return output


class CustomModule2(nn.Module):
    def __init__(self, inputt, hidden, output):
        super(CustomModule2, self).__init__()
        self.linear1 = nn.Linear(inputt, hidden)
        self.linear2 = nn.Linear(hidden, output)
        self.relu = nn.ReLU()

    def forward(self, X):
        hidden = self.relu(self.linear1(X))
        output = self.linear2(hidden)
        return output


class TestApplyHooks:
    config = {"pre_layer_context": {}, "cur_layer_name": "root"}
    models = SimpleModel(10, 10, 10)
    context = {
        "name": "root",
        "submodules": [
            {
                'name': 'module1',
                'is_module_list': True,
                'is_adapt_layer': True,
                'submodules': [
                    {
                        'name': 'module2',
                        "allowed_adapt": True,
                        'submodules': [
                            {
                                "name": "linear1",
                                ModuleAction.RECOMPUTE.name: True
                            },
                            {
                                "name": "linear2",
                                ModuleAction.SWAP.name: True
                            },
                            {
                                "name": "relu",
                            }
                        ]
                    }
                ]
            }
        ]
    }

    target_hook_recompute_module = []
    target_hook_swap_module = []

    def get_module(self, module):
        for sub_module in module:
            if ModuleAction.RECOMPUTE.name in sub_module:
                self.target_hook_recompute_module.append(sub_module[Key.NAME])
            elif ModuleAction.SWAP.name in sub_module:
                self.target_hook_swap_module.append(sub_module[Key.NAME])
            if Key.SUBMODULES in sub_module:
                self.get_module(sub_module[Key.SUBMODULES])

    def get_target_hook_module(self, context):
        self.target_hook_recompute_module = []
        self.target_hook_swap_module = []
        self.get_module(context[Key.SUBMODULES])

        print(f"{self.target_hook_recompute_module=}")
        print(f"{self.target_hook_swap_module=}")

    def test_apply_hooks(self):
        AdaptMemApplyManager().apply_hook_to_model(self.models, self.context, {}, True)
        print(f"{RecomputeHook().recompute_modules=}")
        self.get_target_hook_module(self.context)
        assert len(RecomputeHook().recompute_modules) == len(self.target_hook_recompute_module)


# test apply_op_to_context
parallels = [4, 3, 0]


def get_pp_size():
    return parallels[0]


def get_vpp_size():
    return parallels[1]


def get_pp_rank():
    return parallels[2]


def init_args():
    args = parse_args(None, True)
    set_args(args)
    args.num_layers = 12
    args.micro_batch_size = 2
    args.global_batch_size = 16
    args.pipeline_model_parallel_size = 2
    args.data_parallel_size = 1
    ps.get_pipeline_model_parallel_world_size = get_pp_size
    ps.get_virtual_pipeline_model_parallel_world_size = get_vpp_size
    ps.get_pipeline_model_parallel_rank = get_pp_rank
    try:
        get_num_microbatches()
    except Exception:
        init_num_microbatches_calculator(0, None, args.global_batch_size, args.micro_batch_size,
                                         args.data_parallel_size)


class TestApplyToContext:
    local_adapt_policy_list_normal = [[2, 2, 0, 0, 0, 1, 0, 1]]
    local_adapt_policy_list_full = [[2, 0, 1, 0, 0, 1, 0, 1]]
    local_context = {
        'name': 'root',
        'submodules': [
            {
                'name': 'module',
                'deep': 1,
                'prefix_name': 'module0',
                'is_module_list': True,
                'is_adapt_layer': True,
                'submodules': [
                    {
                        'name': 'experts',
                        'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer',
                        'submodules': [
                            {
                                'name': 'experts',
                                'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts',
                                'submodules': [
                                    {
                                        'name': '0',
                                        'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts',
                                        'submodules': [
                                            {
                                                'name': 'dense_h_to_4h',
                                                'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.0',
                                                'input': 35.203125,
                                                'memory': 176.01611328125,
                                                'forward_cnt': 14,
                                                'pre_total_time': 59.56637954711914,
                                                'avg_time': 4.254741396222796,
                                                'output': 140.8125,
                                                'is_function': True
                                            },
                                            {
                                                'name': 'dense_4h_to_h',
                                                'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.0',
                                                'input': 140.8125,
                                                'memory': 176.01611328125,
                                                'forward_cnt': 14,
                                                'pre_total_time': 62.23174047470093,
                                                'avg_time': 4.4451243196214945,
                                                'output': 35.203125
                                            }
                                        ],
                                        'input': 35.203125,
                                        'memory': 352.03271484375,
                                        'forward_cnt': 14,
                                        'pre_total_time': 126.248459815979,
                                        'avg_time': 9.017747129712786,
                                        'output': 35.203125
                                    },
                                    {
                                        'name': '1',
                                        'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts',
                                        'submodules': [
                                            {
                                                'name': 'dense_h_to_4h',
                                                'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.1',
                                                'input': 35.203125,
                                                'memory': 176.01611328125,
                                                'forward_cnt': 14,
                                                'pre_total_time': 57.57001972198486,
                                                'avg_time': 4.112144265856061,
                                                'output': 140.8125
                                            },
                                            {
                                                'name': 'dense_4h_to_h',
                                                'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.1',
                                                'input': 140.8125,
                                                'memory': 176.01611328125,
                                                'forward_cnt': 14,
                                                'pre_total_time': 62.1959810256958,
                                                'avg_time': 4.442570073263986,
                                                'output': 35.203125
                                            }
                                        ],
                                        'input': 35.203125,
                                        'memory': 352.03271484375,
                                        'forward_cnt': 14,
                                        'pre_total_time': 124.16671848297119,
                                        'avg_time': 8.869051320212227,
                                        'output': 35.203125
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'name': 'experts',
                        'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer',
                        'submodules': [
                            {
                                'name': 'experts',
                                'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts',
                                'submodules': [
                                    {
                                        'name': '0',
                                        'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts',
                                        'submodules': [
                                            {
                                                'name': 'dense_h_to_4h',
                                                'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.0',
                                                'is_function': True
                                            },
                                            {
                                                'name': 'dense_4h_to_h',
                                                'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.0',
                                            }
                                        ],
                                    },
                                    {
                                        'name': '1',
                                        'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts',
                                        'submodules': [
                                            {
                                                'name': 'dense_h_to_4h',
                                                'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.1',
                                            },
                                            {
                                                'name': 'dense_4h_to_h',
                                                'prefix_name': 'module0.module.module.language_model.encoder.layers.0.mlp.block.moe_layer.experts.experts.1',
                                            }
                                        ],
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }

    target_recompute_module_index = []
    target_swap_module_index = []

    def get_target_module_index(self, adapt_policy_list):
        self.target_recompute_module_index = []
        self.target_swap_module_index = []
        adapt_nodes = []
        for policy in adapt_policy_list:
            n = policy[0]
            if policy[1] == LayerAction.FULL_RECOMPUTE:
                status = LayerAction.FULL_RECOMPUTE
                for i in range(0, n):
                    adapt_nodes.extend([status for _ in range(len(policy[2:]))])
            elif policy[1] == LayerAction.FULL_SWAP:
                status = LayerAction.FULL_SWAP
                for i in range(0, n):
                    adapt_nodes.extend([status for _ in range(len(policy[2:]))])
            elif policy[1] == LayerAction.ADAPTIVE:
                for i in range(0, n):
                    adapt_nodes.extend(policy[2:])
        print(f"{adapt_nodes=}")

        for i, nodes in enumerate(adapt_nodes):
            if nodes == ModuleAction.RECOMPUTE:
                self.target_recompute_module_index.append(i)
            elif nodes == ModuleAction.SWAP:
                self.target_swap_module_index.append(i)

        print(f"{self.target_recompute_module_index=}")
        print(f"{self.target_swap_module_index=}")

    def get_real_module_index(self, context, adapt_policy_list):
        apply = AdaptMemApplyManager()

        ordered_layers = []
        apply.get_ordered_layers(context, ordered_layers, True)
        apply.no_adapt_modules = []

        all_ordered_module = []
        idx = 0
        for policy in adapt_policy_list:
            n = policy[0]
            for i in range(idx, idx + n):
                if Key.SUBMODULES not in ordered_layers[i]:
                    continue
                apply.cur_module_index = 0
                ordered_module = []
                apply.get_ordered_modules(ordered_layers[i][Key.SUBMODULES], ordered_module, i)
                all_ordered_module.extend(ordered_module)
            idx += n

        return all_ordered_module

    def compare_module_with_target(self, all_ordered_module):
        cur_recompute_module_index = []
        cur_swap_module_index = []
        for i, module in enumerate(all_ordered_module):
            if Key.IS_FUNCTION in module:
                if i in self.target_recompute_module_index:
                    cur_recompute_module_index.append(i)
                elif i in self.target_swap_module_index:
                    cur_swap_module_index.append(i)

            if ModuleAction.RECOMPUTE.name in module:
                cur_recompute_module_index.append(i)
            elif ModuleAction.SWAP.name in module:
                cur_swap_module_index.append(i)
        print(f"{cur_recompute_module_index=}")
        print(f"{cur_swap_module_index=}")

        assert cur_recompute_module_index == self.target_recompute_module_index
        assert cur_swap_module_index == self.target_swap_module_index

    def test_apply_op_to_context_normal(self):
        init_args()
        apply = AdaptMemApplyManager()
        adapt_policy_list = self.local_adapt_policy_list_normal
        result = apply.apply_op_to_context(adapt_policy_list, self.local_context)
        print(f"{result=}")
        # get target module positions for recompute and swap
        self.get_target_module_index(adapt_policy_list)
        # get current module positions and compare
        all_ordered_module = self.get_real_module_index(result, adapt_policy_list)
        self.compare_module_with_target(all_ordered_module)

    def test_apply_op_to_context_full(self):
        init_args()
        apply = AdaptMemApplyManager()
        adapt_policy_list = self.local_adapt_policy_list_full
        result = apply.apply_op_to_context(adapt_policy_list, self.local_context)
        print(f"{result=}")
        # get target module positions for recompute and swap
        self.get_target_module_index(adapt_policy_list)
        # get current module positions and compare
        all_ordered_module = self.get_real_module_index(result, adapt_policy_list)
        self.compare_module_with_target(all_ordered_module)
