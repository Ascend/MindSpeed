# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import copy

import pytest
import torch
import mindspeed.megatron_adaptor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_npu
from mindspeed.core.memory.compress_dense.compress_tensor import ActivationCompress

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = "npu:0"


class MockArgs:
    def __init__(self):
        self.iteration = 0
        self.curr_iteration = 0
        self.compress_dense = "level1"

    def set_mock_step(self, step):
        self.curr_iteration = step


mock_args = MockArgs()
if_compress = False


class SimpleLayer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLayer, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        global if_compress
        global mock_args
        if if_compress:
            if not hasattr(self, "ac"):
                self.ac = ActivationCompress(mock_args, "simplelayer_ctm")
            self.ac.compress_and_wait_decompress_async_for_previous_layer(x)
        out = self.linear1(x)
        if if_compress:
            self.ac.decompress_and_wait_compress_async_for_previous_layer(out)
        out = F.relu(out)
        if if_compress:
            self.ac.order_record(out)
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


def train_model_with_compress(model, optimizer, input_tensor, target, steps=5):
    global if_compress
    global mock_args
    if_compress = True
    for step in range(steps):
        mock_args.set_mock_step(step)
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
    not hasattr(torch_npu, "npu_hans_encode"),
    reason="Skip UT due to missing torch_npu APIs in CI"
)
class TestCompressTensor:

    def test_CompressTensor(self):
        input_tensor = torch.randn(1024, 64).to(device)
        input_tensor.requires_grad_(True)
        target = torch.randn(1024, 64).to(device)
        model = SimpleModel().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        model_ = copy.deepcopy(model)
        optimizer_ = optim.SGD(model_.parameters(), lr=0.01)
        input_tensor_ = input_tensor.clone()
        input_tensor_.requires_grad_(True)
        target_ = target.clone()

        output, grad = train_model(model, optimizer, input_tensor, target)
        output_, grad_ = train_model_with_compress(model_, optimizer_, input_tensor_, target_)

        assert (torch.allclose(output, output_, atol=1e-6))
        for g1, g2 in zip(grad, grad_):
            assert (torch.allclose(g1, g2, atol=1e-6))
