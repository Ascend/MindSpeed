from math import isclose

import torch
import torch.nn as nn

from mindspeed import megatron_adaptor
from mindspeed.core.memory.adaptive_memory.adaptive_memory_profiling import AdaptiveMemoryProfiling
from mindspeed.core.memory.adaptive_memory.adaptive_memory_tool import AdaptiveStepMgr, ContextKey as Key
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args

from unit_tests.common import DistributedTest


class TwoLayerModel(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(TwoLayerModel, self).__init__()
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.relu = nn.ReLU()

    def forward(self, input_):
        hidden = self.relu(self.fc1(input_))
        output = self.fc2(hidden)
        return output


class TestRecursiveHook(DistributedTest):
    world_size = 1
    args = parse_args(None, True)
    set_args(args)
    prof = AdaptiveMemoryProfiling()
    prof.allowed_adapt_module.append(nn.Linear)

    def test_reset_profiling_hooks_given_sized_profiling_hooks_should_get_empty_profiling_hooks(self):
        model = TwoLayerModel(1024, 1024, 1024).npu()
        input_ = torch.rand((5, 1024, 1024), dtype=torch.float16, device=torch.npu.current_device())
        output_ = []

        def hook_forward_with_input(lst):
            def hook_forward(*args, **kwargs):
                lst.append(1)

            return hook_forward

        handle = model.register_forward_hook(hook_forward_with_input(output_))
        self.prof.profiling_hooks.append(handle)
        assert len(output_) == 0
        assert len(self.prof.profiling_hooks) == 1
        model(input_)
        assert len(output_) == 1
        assert len(self.prof.profiling_hooks) == 1
        self.prof.reset_profiling_hooks()
        model(input_)
        assert len(output_) == 1
        assert len(self.prof.profiling_hooks) == 0

    def test_tag_module_given_manual_context_should_tag_successful(self):
        self._reset_prof_instance()
        ctx = {
            "name": "root",
            "submodules": [
                {
                    "name": "fc1",
                    "prefix_name": "module",
                },
                {
                    "name": "layers",
                    "prefix_name": "module",
                    "submodules": [
                        {
                            "name": "fc2",
                            "prefix_name": "layers"
                        }
                    ]
                }
            ]
        }
        ctx_ans = {
            "name": "root",
            "submodules": [
                {
                    "name": "fc1",
                    "prefix_name": "module",
                    "allowed_adapt": True,
                    "is_adapt_layer": True
                },
                {
                    "name": "layers",
                    "prefix_name": "module",
                    "submodules": [
                        {
                            "name": "fc2",
                            "prefix_name": "layers",
                            "allowed_adapt": True
                        }
                    ],
                    "is_module_list": True,
                    "is_adapt_layer": True
                }
            ]
        }
        ctx_ret_1 = self.prof._tag_module(ctx, ctx[Key.SUBMODULES][0], False, False)
        ctx_ret_2 = self.prof._tag_module(ctx, ctx[Key.SUBMODULES][0], True, False)
        ctx_ret_3 = self.prof._tag_module(ctx[Key.SUBMODULES][1], ctx[Key.SUBMODULES][1][Key.SUBMODULES][0], True, True)
        assert ctx_ret_1 is True
        assert ctx_ret_2 is False
        assert ctx_ret_3 is False
        assert ctx == ctx_ans

    def test_construct_ctx_recursively_given_model_should_get_right_profiling_context(self):
        self._reset_prof_instance()
        ctx = {
            'name': 'root',
            'deep': 0,
            'submodules': [
                {
                    'name': 'fc1',
                    'deep': 1,
                    'prefix_name': 'module',
                    'allowed_adapt': True,
                    'is_adapt_layer': True
                },
                {
                    'name': 'fc2',
                    'deep': 1,
                    'prefix_name': 'module',
                    'allowed_adapt': True,
                    'is_adapt_layer': True
                },
                {
                    'name': 'relu',
                    'deep': 1,
                    'prefix_name': 'module'
                }
            ]
        }

        model = TwoLayerModel(1024, 1024, 1024).npu()
        self.prof.construct_ctx_recursively(1, Key.MODULE, model, self.prof.context, True)
        assert ctx == self.prof.context

    def test_record_submodule_forward_time_given_module_should_record_time_successful_at_profiling_step(self):
        model = TwoLayerModel(1024, 1024, 1024).npu()
        input_ = torch.rand((5, 1024, 1024), dtype=torch.float16, device=torch.npu.current_device())
        self._reset_prof_instance()
        AdaptiveStepMgr().reset_step(5)
        self.prof.construct_ctx_recursively(1, Key.MODULE, model, self.prof.context, True)
        self.prof.register_hook_recursively(model, self.prof.context, in_first_module=True)

        model(input_)
        assert self.prof.time_event_list is not None
        self.prof.record_time()
        for ctx in self.prof.context[Key.SUBMODULES]:
            assert ctx[Key.FORWARD_CNT] == 1
            assert ctx[Key.PRE_TOTAL_TIME] > 0
            assert ctx[Key.AVG_TIME] == ctx[Key.PRE_TOTAL_TIME]

        model(input_)
        self.prof.record_time()
        for ctx in self.prof.context[Key.SUBMODULES]:
            assert ctx[Key.FORWARD_CNT] == 2
            assert ctx[Key.PRE_TOTAL_TIME] > 0
            assert ctx[Key.AVG_TIME] < ctx[Key.PRE_TOTAL_TIME]

    def test_cal_input_output_size_given_10_megabyte_tensor_should_return_10_megabyte(self):
        t_ = torch.rand((5, 1024, 1024), dtype=torch.float16)
        t_size = self.prof.cal_input_output_size(t_) / 1024 / 1024
        assert isclose(t_size, 10)

    def test_cal_input_output_size_given_10_megabyte_tensor_list_should_return_10_megabyte(self):
        t_list = [torch.rand((2, 1024, 1024), dtype=torch.float16),
                  torch.rand((3, 1024, 1024), dtype=torch.float16)]
        t_list_size = self.prof.cal_input_output_size(t_list) / 1024 / 1024
        assert isclose(t_list_size, 10)

    def test_cal_input_output_size_given_10_megabyte_nested_tensor_list_should_return_10_megabyte(self):
        t_nested_list = [torch.rand((2, 1024, 1024), dtype=torch.float16),
                         [torch.rand((1, 1024, 1024), dtype=torch.float16),
                          torch.rand((1, 1024, 1024), dtype=torch.float16),
                          torch.rand((1, 1024, 1024), dtype=torch.float16)]]
        t_nested_list_size = self.prof.cal_input_output_size(t_nested_list) / 1024 / 1024
        assert isclose(t_nested_list_size, 10)

    def test_register_hook_given_model_should_input_add_output_less_than_memory_at_stop_profiling_step(self):
        model = TwoLayerModel(1024, 1024, 1024).npu()
        input_ = torch.rand((5, 1024, 1024), dtype=torch.float16, device=torch.npu.current_device())
        self._reset_prof_instance()
        AdaptiveStepMgr().reset_step(10)
        self.prof.construct_ctx_recursively(1, Key.MODULE, model, self.prof.context, True)

        index = 0
        for module in model.children():
            if Key.SUBMODULES not in self.prof.context:
                continue

            current_ctx = self.prof.context[Key.SUBMODULES][index]
            name = current_ctx[Key.NAME]
            prefix_name = current_ctx[Key.PREFIX_NAME]
            self.prof._register_hook(module, prefix_name, name, current_ctx)
            index += 1

        model(input_)
        for subcontext in self.prof.context[Key.SUBMODULES]:
            assert subcontext[Key.INPUT] >= 0
            assert subcontext[Key.OUTPUT] >= 0
            assert subcontext[Key.INPUT] + subcontext[Key.OUTPUT] <= subcontext[Key.MEMORY]

    def _reset_prof_instance(self):
        self.prof.context = {'name': 'root', 'deep': 0, 'submodules': []}
        self.prof.forward_time = 0
        self.prof.profiling_hooks = []
        self.prof.time_event_list = []
        self.prof.checkpointed_modules = []
        AdaptiveStepMgr().skip_steps = 3
        AdaptiveStepMgr().recompute_profiling_steps = 7


class TestFunctionHook:
    def test_insert_function_no_child(self):
        ctx = {
            "module": [],
            "deep": 0,
            "submodules": [
                {
                    "name": "fc_a",
                    "deep": 1,
                    "prefix_name": "module",
                    "allowed_adapt": True,
                    "is_adapt_layer": True
                },
                {
                    "name": "fc_b",
                    "deep": 1,
                    "prefix_name": "module",
                    "allowed_adapt": True,
                    "is_adapt_layer": True,
                    "submodules": [
                        {
                            "name": "child_a",
                            "deep": 2,
                            "prefix_name": "module.fc_b",
                            "submodules": [
                                {
                                    "name": "child_a_a",
                                    "deep": 3,
                                    "prefix_name": "module.fc_b.child_a"
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        profiling = AdaptiveMemoryProfiling()
        profiling.context = ctx

        new_ctx = {
            "name": "PipeExpert",
            "deep": 3,
            "prefix_name": "module.fc_b.child_a",
            "input": 8.8125,
            "memory": 132.1972740292549,
            "pre_total_time": 13.211959838867188,
            "output": 8.8125
        }

        profiling.insert_func_profiling(new_ctx, "")
        parent = profiling.context["submodules"][1]["submodules"][0]
        siblings = parent["submodules"]
        found = next((x for x in siblings if x["name"] == "PipeExpert"), None)
        assert found

    def test_insert_function_with_child(self):
        ctx = {
            "module": [],
            "deep": 0,
            "submodules": [
                {
                    "name": "fc_a",
                    "deep": 1,
                    "prefix_name": "module",
                    "allowed_adapt": True,
                    "is_adapt_layer": True
                },
                {
                    "name": "fc_b",
                    "deep": 1,
                    "prefix_name": "module",
                    "allowed_adapt": True,
                    "is_adapt_layer": True,
                    "submodules": [
                        {
                            "name": "child_a",
                            "deep": 2,
                            "prefix_name": "module.fc_b",
                            "submodules": [
                                {
                                    "name": "child_a_a",
                                    "deep": 3,
                                    "prefix_name": "module.fc_b.child_a"
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        profiling = AdaptiveMemoryProfiling()
        profiling.context = ctx

        new_ctx = {
            "name": "PipeExpert",
            "deep": 2,
            "prefix_name": "module.fc_b",
            "input": 8.8125,
            "memory": 132.1972740292549,
            "pre_total_time": 13.211959838867188,
            "output": 8.8125
        }

        profiling.insert_func_profiling(new_ctx, "child_a")
        parent = profiling.context["submodules"][1]
        siblings = parent["submodules"]
        found = next((x for x in siblings if x["name"] == "PipeExpert"), None)
        assert found
