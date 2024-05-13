import torch
from ascendspeed import megatron_adaptor
import torch.nn as nn
from ascendspeed.core.memory.adaptive_recomputing.adaptive_recompute import get_adaptive_recomputing
from ascendspeed.core.memory.adaptive_recomputing.adaptive_recompute_apply import get_recompute_hook
from ascendspeed.core.memory.adaptive_recomputing.adaptive_recompute import allowed_recomputing_module_wrapper

from unit_tests.common import DistributedTest


class MockArgs:
    mock_args = None

    def __init__(self):
        self.adaptive_recompute_device_size = -1


class TwoLayerModel(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(TwoLayerModel, self).__init__()
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.relu = nn.ReLU()

    def forward(self, X):
        hidden = self.relu(self.fc1(X))
        output = self.fc2(hidden)
        return output


class TestRecursiveHook(DistributedTest):
    world_size = 1

    def test_allowed_recomputing_module_wrapper(self):
        recomputing = get_adaptive_recomputing()
        assert len(recomputing.allowed_recomputing_module) == 0
        allowed_recomputing_module_wrapper(nn.Linear)
        assert len(recomputing.allowed_recomputing_module) == 1

    def test_construct_context_recursive(self):
        model = TwoLayerModel(10, 10, 10)
        recomputing = get_adaptive_recomputing()
        allowed_recomputing_module_wrapper(nn.Linear)
        recomputing.construct_context_recursive("module", model, recomputing.context, True)

    def test_register_recursive_hook(self):
        model = TwoLayerModel(10, 10, 10)
        ctx = {
            "module":[],
            "layers":[
                {
                    "name": "fc1",
                    "prefix_name": "module",
                    "allowed_recomputing": True,
                    "is_recomputing_layer": True
                },
                {
                    "name": "fc2",
                    "prefix_name": "module",
                    "allowed_recomputing": True,
                    "is_recomputing_layer": True
                },
                {
                    "name": "relu",
                    "prefix_name": "module"
                }
            ]
        }

        recomputing = get_adaptive_recomputing()
        recomputing.set_profiling_step(10)
        assert len(recomputing.modules_hooks) == 0
        recomputing.register_recursive_hook(model, ctx, recomputing.profiling_prefix)

        assert len(recomputing.modules_hooks) != 0

    def test_pre_hook_func(self):
        model = TwoLayerModel(10, 10, 10)
        ctx = {}
        recomputing = get_adaptive_recomputing()
        recomputing.set_profiling_step(10)
        prefix_name = "module"

        for name, module in model.named_children():
            if str.isdigit(name) and name != "0":
                module.no_checkpoint_forward = module.forward
                module.forward = get_recompute_hook().hook_checkpoint_forward(module.forward)
                self.checkpointed_modules.append(module)
            if 'layers' not in ctx:
                ctx['layers'] = []
            current_ctx = {}

            current_ctx['name'] = name
            current_ctx['prefix_name'] = prefix_name
            if 'layers' in ctx:
                ctx['layers'].append(current_ctx)

            if 'module' in recomputing.context:
                recomputing.context['module'].append(current_ctx)

            assert 'memory' not in current_ctx
            recomputing.pre_hook_func(current_ctx, prefix_name, name)
            assert current_ctx['memory'] is not None

    def test_post_hook_func(self):
        model = TwoLayerModel(10, 10, 10)
        ctx = {}
        recomputing = get_adaptive_recomputing()
        recomputing.set_profiling_step(10)
        prefix_name = "module"

        for name, module in model.named_children():
            if str.isdigit(name) and name != "0":
                module.no_checkpoint_forward = module.forward
                module.forward = get_recompute_hook().hook_checkpoint_forward(module.forward)
                self.checkpointed_modules.append(module)
            if 'layers' not in ctx:
                ctx['layers'] = []
            current_ctx = {}

            current_ctx['name'] = name
            current_ctx['prefix_name'] = prefix_name
            if 'layers' in ctx:
                ctx['layers'].append(current_ctx)

            if 'module' in recomputing.context:
                recomputing.context['module'].append(current_ctx)
            recomputing.pre_hook_func(current_ctx, prefix_name, name)

            assert 'peak_memory' not in current_ctx
            args_tensor = torch.rand((10, 10, 10), dtype=torch.float16, requires_grad=True, device=torch.npu.current_device())
            output_tensor = torch.rand((10, 10, 10), dtype=torch.float16, requires_grad=True, device=torch.npu.current_device())
            recomputing.post_hook_func(current_ctx, prefix_name, name, args_tensor, output_tensor)
            assert current_ctx['peak_memory'] is not None
