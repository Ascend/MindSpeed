from unittest import mock

import torch
from torch import nn
import pytest
from mindspeed import megatron_adaptor

from megatron.core.tensor_parallel import checkpoint
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from tests_extend.unit_tests.common import DistributedTest


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


class DummyCheckpointManager:
    def __init__(self, open_ri_pipe=False, enable_independent_recompute=False, enable_recompute=True):
        self.open_ri_pipe = open_ri_pipe
        self.do_pre_recompute = enable_independent_recompute
        self.checkpoint_list = []
        self.chunk_do_recompute = enable_recompute

    def add_recompute(self, recompute_func):
        self.checkpoint_list.append(recompute_func)


class TestCheckpointFunction(DistributedTest):
    world_size = 1
    RiPipeSchedulesFeatureTset().ripipe_schedule_patch()

    layer = nn.Sequential(nn.Linear(4, 5), nn.ReLU())

    def forward(self, x):
        return self.layer(x)

    def test_origin_checkpoint_function(self):
        with mock.patch('mindspeed.core.tensor_parallel.random.get_pipeline_checkpoint_manager') as get_ckpt_manager:
            get_ckpt_manager.return_value = DummyCheckpointManager(open_ri_pipe=False)
            mock_forward = mock.Mock(wraps=self.forward)
            input_data = torch.randn(3, 4, requires_grad=True)
            output = checkpoint(mock_forward, False, input_data)
            assert 'CheckpointFunctionBackward' in str(output.grad_fn)

            assert mock_forward.call_count == 1
            output.sum().backward()
            assert mock_forward.call_count == 2

    def test_disable_recompute(self):
        with mock.patch('mindspeed.core.tensor_parallel.random.get_pipeline_checkpoint_manager') as get_ckpt_manager:
            get_ckpt_manager.return_value = DummyCheckpointManager(open_ri_pipe=True, enable_recompute=False)
            mock_forward = mock.Mock(wraps=self.forward)
            input_data = torch.randn(3, 4, requires_grad=True)
            output = checkpoint(mock_forward, False, input_data)
            assert 'ReluBackward' in str(output.grad_fn)

            assert mock_forward.call_count == 1
            output.sum().backward()
            assert mock_forward.call_count == 1


    def test_new_checkpoint_function(self):
        with mock.patch('mindspeed.core.tensor_parallel.random.get_pipeline_checkpoint_manager') as get_ckpt_manager:
            get_ckpt_manager.return_value = DummyCheckpointManager(open_ri_pipe=True, enable_recompute=True)
            mock_forward = mock.Mock(wraps=self.forward)
            input_data = torch.randn(3, 4, requires_grad=True)
            output = checkpoint(mock_forward, False, input_data)
            assert 'CheckpointFunctionRipipeBackward' in str(output.grad_fn)

            expected_output = self.forward(input_data)
            assert torch.allclose(output, expected_output)

            assert mock_forward.call_count == 1
            output.sum().backward()
            assert mock_forward.call_count == 2

            input_grad = input_data.grad.clone().detach()
            input_data.grad.zero_()

            expected_output.sum().backward()
            assert torch.allclose(input_grad, input_data.grad)


    def test_independent_recompute(self):
        with mock.patch('mindspeed.core.tensor_parallel.random.get_pipeline_checkpoint_manager') as get_ckpt_manager:
            ckpt_manager = DummyCheckpointManager(open_ri_pipe=True, enable_independent_recompute=True, enable_recompute=True)
            get_ckpt_manager.return_value = ckpt_manager
            mock_forward = mock.Mock(wraps=self.forward)
            input_data = torch.randn(3, 4, requires_grad=True)
            output = checkpoint(mock_forward, False, input_data)
            assert 'CheckpointFunctionRipipeBackward' in str(output.grad_fn)

            expected_output = self.forward(input_data)
            assert torch.allclose(output, expected_output)

            assert mock_forward.call_count == 1
            for recompute_func in ckpt_manager.checkpoint_list:
                recompute_func()
            assert mock_forward.call_count == 2
            output.sum().backward()
            assert mock_forward.call_count == 2

            input_grad = input_data.grad.clone().detach()
            input_data.grad.zero_()

            expected_output.sum().backward()
            assert torch.allclose(input_grad, input_data.grad)
