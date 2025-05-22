# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import sys

sys.argv.append("--tp-2d")
from mindspeed import megatron_adaptor_v2

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from megatron.core.parallel_state import destroy_model_parallel, initialize_model_parallel
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel import mappings
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args

from mindspeed.core.tensor_parallel.comm_autograd_function import auto_grad_scatter_along_first_dim, \
    auto_grad_scatter_along_last_dim, auto_grad_sync_gather_along_first_dim, auto_grad_sync_gather_along_last_dim
from mindspeed.core.tensor_parallel.comm_utils import sync_reduce_scatter_along_first_dim
from mindspeed.core.tensor_parallel.tp_2d.layernorm_2d import LayerNorm2D
from mindspeed.core.tensor_parallel.tp_2d.group_api_2d import TPXCollectiveComm, TPYCollectiveComm

from tests_extend.unit_tests.common import DistributedTest
from tests_extend.commons import set_random_seed

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
# Setting sys.argv is mainly to ensure that --tp-2d is not None,
# so that the code block (which will be executed
# after determining that noop_layers is not None)
# will be executed in megatron_adaptor.

sys.argv.remove('--tp-2d')


# a simple model containing LayerNorm
class SimpleModel(nn.Module):
    def __init__(self, norm_size):
        super(SimpleModel, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=norm_size)

    def forward(self, x):
        output = self.layer_norm(x)
        return output


class _OnlyAllGatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_):
        return mappings._gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return mappings._gather_along_last_dim(input_,)

    @staticmethod
    def backward(ctx, grad_output):
        return mappings._split_along_last_dim(grad_output)


def only_all_gather_last_dim_from_tensor_parallel_region(input_):
    return _OnlyAllGatherFromTensorParallelRegion.apply(input_)


class SimpleDistModel(nn.Module):
    def __init__(self, norm_size):
        super(SimpleDistModel, self).__init__()
        last_dim_split_comm_intf = TPYCollectiveComm()
        self.layer_norm = LayerNorm2D(norm_size, last_dim_split_comm_intf=last_dim_split_comm_intf)

    def forward(self, x):
        output = self.layer_norm(x)
        return output


class TestLayernorm2dRsFirstDim(DistributedTest):
    world_size = 4
    diff_value = 1e-5
    reuse_dist_env = False

    @staticmethod
    def get_layernorm_grad(dist_schedule, h, input_x, targets):
        # some random input data and target output
        input_x_data = input_x.clone().detach()
        targets_data = targets.clone().detach()
        input_x_data.requires_grad_()
        # model instance
        if dist_schedule:
            model = SimpleDistModel(h).npu()
        else:
            model = SimpleModel(h).npu()

        # loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        # forward, predict
        if dist_schedule:
            # s,b,E -> s/x,b,h/y
            input_x_data = auto_grad_scatter_along_first_dim(input_x_data, TPXCollectiveComm)
            input_x_data = auto_grad_scatter_along_last_dim(input_x_data, TPYCollectiveComm)
            outputs = model(input_x_data)
            outputs = auto_grad_sync_gather_along_first_dim(outputs, TPXCollectiveComm)
            outputs = auto_grad_sync_gather_along_last_dim(outputs, TPYCollectiveComm)
        else:
            outputs = model(input_x_data)

        # loss
        loss = criterion(outputs, targets_data)
        # backward, grad
        loss.backward()
        return outputs, model.layer_norm.weight.grad, model.layer_norm.bias.grad

    def test_layer_norm_should_be_allclose_give_same_input_when_norm_1d_2d(self):
        set_random_seed(1)
        args = parse_args(None, True)
        set_args(args)
        tp, tp_x, tp_y = 4, 2, 2
        pp = self.world_size // tp
        h = 8
        seq = 16
        b = 1
        args.tp_x = tp_x
        args.tp_y = tp_y
        # generate some random input data and target output.
        input_x = torch.randn(seq, b, h, requires_grad=True).npu()
        targets = torch.randn(seq, b, h).npu()
        destroy_model_parallel()
        initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_split_rank=None)
        # 2d layer_norm
        output_2d, weight_grad_2d, bias_grad_2d = self.get_layernorm_grad(dist_schedule=1, h=h, input_x=input_x, targets=targets)
        # 1d layer_norm
        output_1d, weight_grad_1d, bias_grad_1d = self.get_layernorm_grad(dist_schedule=0, h=h, input_x=input_x, targets=targets)
        assert torch.allclose(output_2d, output_1d, rtol=self.diff_value, atol=self.diff_value)
        weight_grad_1d = tensor_parallel.scatter_to_tensor_model_parallel_region(weight_grad_1d)
        bias_grad_1d = tensor_parallel.scatter_to_tensor_model_parallel_region(bias_grad_1d)
        weight_grad_2d = sync_reduce_scatter_along_first_dim(weight_grad_2d, TPXCollectiveComm)
        bias_grad_2d = sync_reduce_scatter_along_first_dim(bias_grad_2d, TPXCollectiveComm)
        assert torch.allclose(weight_grad_2d, weight_grad_1d, rtol=self.diff_value, atol=self.diff_value)
        assert torch.allclose(bias_grad_2d, bias_grad_1d, rtol=self.diff_value, atol=self.diff_value)
