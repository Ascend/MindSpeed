from math import isclose
from copy import deepcopy

import torch
import torch.nn as nn
from mindspeed import megatron_adaptor
from mindspeed.core.memory.adaptive_memory.adaptive_memory_prefetch import AdaptiveMemoryPrefetch
from mindspeed.core.memory.adaptive_memory.adaptive_memory_swap_manager import SwapManager
from mindspeed.core.memory.adaptive_memory.adaptive_memory_tool import AdaptiveStepMgr
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args

from unit_tests.common import DistributedTest


ctx = {
    "name": "module0",
    "deep": 0,
    "prefix_name": "module",
    "submodules": [
        {
            "name": "layer0",
            "deep": 1,
            "prefix_name": "module.module0",
            "submodules": [
                {
                    "name": "fc1",
                    "deep": 2,
                    "prefix_name": "module.module0.layer0",
                    "is_modlue_of_layer0": True,
                },
                {
                    "name": "fc2",
                    "deep": 2,
                    "prefix_name": "module.module0.layer0",
                    "is_modlue_of_layer0": True,
                },
                {
                    "name": "relu",
                    "deep": 2,
                    "prefix_name": "module.module0.layer0",
                    "is_modlue_of_layer0": True,
                }
            ],
            "is_layer0_of_module0": True,
            "is_modlue_of_layer0": True
        },
        {
            "name": "layer1",
            "deep": 1,
            "prefix_name": "module.module0",
            "submodules": [
                {
                    "name": "fc1",
                    "deep": 2,
                    "prefix_name": "module.module0.layer1"
                },
                {
                    "name": "fc2",
                    "deep": 2,
                    "prefix_name": "module.module0.layer1"
                },
                {
                    "name": "relu",
                    "deep": 2,
                    "prefix_name": "module.module0.layer1"
                }
            ]
        }
    ]
}


class TwoLayerModel(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(TwoLayerModel, self).__init__()
        self.layer0 = LayerModel(num_input, num_hidden, num_output)
        self.layer1 = LayerModel(num_input, num_hidden, num_output)

    def forward(self, input_):
        out1 = self.layer0(input_)
        out = self.layer1(out1)
        return out


class LayerModel(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(LayerModel, self).__init__()
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.relu = nn.ReLU()

    def forward(self, input_):
        hidden = self.relu(self.fc1(input_))
        output = self.fc2(hidden)
        return output


class TestAdaptiveMemoryPrefetch(DistributedTest):
    args = parse_args(None, True)
    set_args(args)
    adaptive_memory_prefetch = AdaptiveMemoryPrefetch()

    def _reset_parameters(self):
        self.adaptive_memory_prefetch.chunk_num = 1
        self.adaptive_memory_prefetch.prefetch_deep_list = [1]
        self.adaptive_memory_prefetch.prefetch_hook_interval = len(self.adaptive_memory_prefetch.prefetch_deep_list)
        self.adaptive_memory_prefetch.config = {
            "pre_layer_full_name": "",
            "cur_layer_name": "module",
        }
        AdaptiveStepMgr().cur_step = 13
        AdaptiveStepMgr().skip_steps = 3
        AdaptiveStepMgr().recompute_profiling_steps = 5
        AdaptiveStepMgr().layer_profiling_steps = 5

    def test_register_hook_for_swap_prof(self):
        # 初始化参数
        context = deepcopy(ctx)
        model = TwoLayerModel(1024, 1024, 1024).npu()
        input_ = torch.rand((5, 1024, 1024), dtype=torch.float16, device=torch.npu.current_device())
        self._reset_parameters()

        # 验证swap profiling中能否正确添加hook
        self.adaptive_memory_prefetch.register_recursive_apply_prefetch(self.adaptive_memory_prefetch.config, model, context, True)
        assert len(self.adaptive_memory_prefetch.modules_hooks) > 0
        assert len(SwapManager().prefetch_hooked_modules) > 0
        assert len(SwapManager().post_layer_forward_and_pre_layer_backward_hooks) > 0

        # 验证swap profiling中添加的hook能否达到预期
        model(input_)
        assert len(self.adaptive_memory_prefetch.prefetch_module_event_dict) != 0

        # 验证reset功能
        self.adaptive_memory_prefetch.reset_adaptive_prefetch_all_hooks()
        assert len(self.adaptive_memory_prefetch.modules_hooks) == 0
        assert len(SwapManager().prefetch_hooked_modules) == 0
        assert len(SwapManager().post_layer_forward_and_pre_layer_backward_hooks) == 0

    def test_register_hook_for_swap_adjust(self):
        # 初始化参数
        self._reset_parameters()
        context = deepcopy(ctx)
        model = TwoLayerModel(1024, 1024, 1024).npu()

        # 验证swap 微调中能否正确添加hook
        self.adaptive_memory_prefetch.need_swap_module_name.append("module.module0.layer0.fc1")
        self.adaptive_memory_prefetch.need_swap_module_ctx.append(context['submodules'][0]['submodules'][0])
        self.adaptive_memory_prefetch.register_recursive_apply_prefetch(self.adaptive_memory_prefetch.config, model, context, False)
        assert len(SwapManager().prefetch_hooked_modules) > 0
        assert len(SwapManager().post_layer_forward_and_pre_layer_backward_hooks) > 0