# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import argparse
import copy
import os
from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

import mindspeed.megatron_adaptor_v2
from mindspeed.core.optimizer.adamw import AdamW
from mindspeed.core.optimizer.virtual_optimizer.virtual_adam import virtual_optimizer_step_impl, VirtualAllocator


def virtual_optimizer_step(self, closure=None):
    if not hasattr(self, "virtual_allocator"):
        self.virtual_allocator = VirtualAllocator(0, 1, [1.0])
    if not hasattr(self, "print_swap_flag"):
        self.print_swap_flag = False
    with torch.no_grad():
        loss = virtual_optimizer_step_impl(self, closure)
    return loss


class SimpleLayer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLayer, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        return out


class SimpleModel(nn.Module):

    def __init__(self, input_size=64, hidden_size=128, output_size=64):
        super(SimpleModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dense_layer = SimpleLayer(input_size, hidden_size, output_size)
        self.model = nn.Sequential(self.dense_layer, self.dense_layer, self.dense_layer)

    def forward(self, x):
        out = self.model(x)
        return out


def train_model(model, optimizer, input_tensor, target, steps=5):
    for _ in range(steps):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
    expect_grad = []
    for _, param in model.named_parameters():
        if param.grad is not None:
            expect_grad.append(param.grad)
    return output, expect_grad


@pytest.mark.skipif(
    not hasattr(torch_npu, "empty_with_swapped_memory"),
    reason="Skip UT due to missing torch_npu APIs in CI"
)
class TestVirtualOptimizer:

    def test_VirtualOptimizer(self):
        input_tensor = torch.randn(1024, 64).to("npu:0")
        input_tensor.requires_grad_(True)
        target = torch.randn(1024, 64).to("npu:0")
        model = SimpleModel().to("npu:0")
        optimizer = AdamW(model.parameters(), lr=0.01)

        AdamW.step = virtual_optimizer_step
        model_ = copy.deepcopy(model)
        optimizer_ = AdamW(model_.parameters(), lr=0.01)
        input_tensor_ = input_tensor.clone()
        input_tensor_.requires_grad_(True)
        target_ = target.clone()

        output, grad = train_model(model, optimizer, input_tensor, target)
        output_, grad_ = train_model(model_, optimizer_, input_tensor_, target_)

        assert (torch.allclose(output, output_, atol=1e-6))
        for g1, g2 in zip(grad, grad_):
            assert (torch.allclose(g1, g2, atol=1e-6))
