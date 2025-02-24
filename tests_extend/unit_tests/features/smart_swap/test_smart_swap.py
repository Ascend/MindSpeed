# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import pytest
import torch
from torch.optim.adam import Adam
import torch_npu

from mindspeed.core.memory.smart_swap.swap_adaptor import change_allocator
from mindspeed.core.memory.smart_swap.swap_policy_config import swap_policy_config


@pytest.fixture(scope="class")
def switch_to_pluggable_allocator():
    change_allocator()


@pytest.mark.skip(reason='some torch_npu api not supported in CI, skip this UT!')
@pytest.mark.usefixtures("switch_to_pluggable_allocator")
class TestSmartSwap:

    def init_swap_manager(self):

        def custom_num_micro_batch_fcn():
            return 1

        def custom_get_optimizer_tensors_fcn(optimizer):
            results = []
            for group in optimizer.param_groups:
                amsgrad = group["amsgrad"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    results.append(p.data)

                    state = optimizer.state[p]
                    if len(state) > 0:
                        results.append(state["exp_avg"])
                        results.append(state["exp_avg_sq"])
                        if amsgrad:
                            results.append(state["max_exp_avg_sq"])

            return results

        def custom_get_shared_tensors_fcn(models):
            results = []
            return results

        num_layers = 12
        models = [torch.nn.Transformer(d_model=512, nhead=16, num_encoder_layers=num_layers).npu()]
        optimizer = Adam(models[0].parameters())

        from mindspeed.core.memory.smart_swap.swap_manager import SwapManager
        swap_manager = SwapManager(
            custom_num_micro_batch_fcn,
            models,
            num_layers,
            optimizer=optimizer,
            get_optimizer_tensors_fcn=custom_get_optimizer_tensors_fcn,
            get_shared_tensors_fcn=custom_get_shared_tensors_fcn,
        )

        return models, optimizer, swap_manager

    def test_swap_manager(self):
        from mindspeed.core.memory.smart_swap.swap_manager import SwapRunningStage

        src = torch.rand((512, 32, 512)).npu()
        tgt = torch.rand((1024, 32, 512)).npu()

        models, optimizer, swap_manager = self.init_swap_manager()
        assert swap_manager.running_stage == SwapRunningStage.WARMUP_STAGE
        assert swap_manager.config.enable_profiler == True
        assert swap_manager.config.enable_executor == False

        swap_policy_config.warmup_step = 2
        swap_policy_config.stable_step = 4
        swap_policy_config.reduction_memory = 1 * (1 << 30)

        for iteration in range(3):
            out = models[0](src, tgt)
            loss = out.sum()
            loss.backward()
            optimizer.step()

            if iteration == 0:
                assert swap_manager.config.step == 0
                assert swap_manager.running_stage == SwapRunningStage.WARMUP_STAGE
                assert swap_manager.config.enable_profiler == True
                assert swap_manager.config.enable_executor == False
            if iteration == 1:
                assert swap_manager.config.step == 1
                assert swap_manager.running_stage == SwapRunningStage.WARMUP_STAGE
                assert swap_manager.config.enable_profiler == True
                assert swap_manager.config.enable_executor == False
            if iteration == 2:
                assert swap_manager.config.step == 2
                assert swap_manager.running_stage == SwapRunningStage.SEARCHING_POLICY_STAGE
                assert swap_manager.config.enable_profiler == True
                assert swap_manager.config.enable_executor == True
            if iteration == 3:
                assert swap_manager.config.step == 3
                assert swap_manager.running_stage == SwapRunningStage.SEARCHING_POLICY_STAGE
                assert swap_manager.config.enable_profiler == True
                assert swap_manager.config.enable_executor == True
            if iteration == 4:
                assert swap_manager.config.step == 4
                assert swap_manager.running_stage == SwapRunningStage.STABLE_STAGE
                assert swap_manager.config.enable_profiler == False
                assert swap_manager.config.enable_executor == True

            swap_manager.step()

            if iteration == 0:
                assert swap_manager.config.step == 1
                assert swap_manager.running_stage == SwapRunningStage.WARMUP_STAGE
                assert swap_manager.config.enable_profiler == True
                assert swap_manager.config.enable_executor == False
            if iteration == 1:
                assert swap_manager.config.step == 2
                assert swap_manager.running_stage == SwapRunningStage.SEARCHING_POLICY_STAGE
                assert swap_manager.config.enable_profiler == True
                assert swap_manager.config.enable_executor == True
            if iteration == 2:
                assert swap_manager.config.step == 3
                assert swap_manager.running_stage == SwapRunningStage.SEARCHING_POLICY_STAGE
                assert swap_manager.config.enable_profiler == True
                assert swap_manager.config.enable_executor == True
            if iteration == 3:
                assert swap_manager.config.step == 4
                assert swap_manager.running_stage == SwapRunningStage.STABLE_STAGE
                assert swap_manager.config.enable_profiler == False
                assert swap_manager.config.enable_executor == True
            if iteration == 4:
                assert swap_manager.config.step == 5
                assert swap_manager.running_stage == SwapRunningStage.STABLE_STAGE
                assert swap_manager.config.enable_profiler == False
                assert swap_manager.config.enable_executor == True
