import pytest
import torch
from torch import nn
from mindspeed import megatron_adaptor
from mindspeed.core.tensor_parallel.checkpoint_manager import get_pipeline_checkpoint_manager

from megatron.core.tensor_parallel import checkpoint
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args

from tests_extend.unit_tests.common import DistributedTest


def checkpointed_forward(function, *args):
    return checkpoint(function, False, *args)


class RiPipeSchedulesFeatureTset:
    @staticmethod
    def ripipe_register_patches(patch_manager, args):
        if not args.recompute_in_bubble and not args.recompute_in_advance:
            return
        from mindspeed.core.tensor_parallel.random import checkpoint_wrapper
        from mindspeed.core.memory.common import linear_forward_main_grad_wrapper, linear_backward_main_grad_wrapper
        patch_manager.register_patch('megatron.core.tensor_parallel.random.checkpoint', checkpoint_wrapper)
        from mindspeed.core.pipeline_parallel.ripipe_schedules import get_forward_backward_func_ripipe_patch
        patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                                     get_forward_backward_func_ripipe_patch, force_patch=True)
        patch_manager.register_patch(
            'megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.forward',
            linear_forward_main_grad_wrapper)
        patch_manager.register_patch(
            'megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
            linear_backward_main_grad_wrapper)

    def ripipe_schedule_patch(self):
        args = parse_args(None, True)
        set_args(args)
        args.recompute_in_bubble = True

        from mindspeed.patch_utils import MindSpeedPatchesManager as pm
        self.ripipe_register_patches(pm, args)
        pm.apply_patches()


class TestCheckpointManager(DistributedTest):
    world_size = 1
    RiPipeSchedulesFeatureTset().ripipe_schedule_patch()

    def test_checkpoint_manager(self):
        layer1 = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        layer2 = nn.Sequential(nn.Linear(20, 30), nn.ReLU())
        layer3 = nn.Sequential(nn.Linear(30, 20), nn.ReLU())
        layer4 = nn.Sequential(nn.Linear(20, 10), nn.ReLU())

        input_data = torch.randn(20, 10, requires_grad=True)
        expected_output = layer4(layer3(layer2(layer1(input_data))))

        manager = get_pipeline_checkpoint_manager(num_of_chunks=1)
        manager.open_ri_pipe = True
        manager.do_pre_recompute = True

        output = checkpointed_forward(layer1, input_data)
        manager.disable_recompute()
        output = checkpointed_forward(layer2, output)
        manager.enable_recompute()
        output = checkpointed_forward(layer3, output)
        manager.disable_recompute()
        output = checkpointed_forward(layer4, output)

        with pytest.raises(RuntimeError):
            assert manager.iter_fin()

        manager.batch_fin(0)

        assert torch.allclose(expected_output, output)

        expected_output.sum().backward()
        expected_grad = input_data.grad.clone().detach()
        input_data.grad.zero_()

        manager.recompute_next(0)
        output.sum().backward()
        manager.iter_fin()
        assert torch.allclose(expected_grad, input_data.grad)
