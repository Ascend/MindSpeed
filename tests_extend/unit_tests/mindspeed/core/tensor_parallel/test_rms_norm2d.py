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
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unit_tests.common import DistributedTest
from commons import set_random_seed

from mindspeed import megatron_adaptor

from megatron.core.parallel_state import destroy_model_parallel, initialize_model_parallel
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel import mappings
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.legacy.model import RMSNorm
from mindspeed.core.tensor_parallel.comm_autograd_function import auto_grad_scatter_along_first_dim, \
    auto_grad_scatter_along_last_dim, auto_grad_sync_gather_along_first_dim, auto_grad_sync_gather_along_last_dim
from mindspeed.core.tensor_parallel.comm_utils import sync_reduce_scatter_along_first_dim

from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm, TPYCollectiveComm
from mindspeed.core.tensor_parallel.tp_2d.rms_norm_2d import RMSNorm2D


class SimpleRMSNormModel(nn.Module):
    def __init__(self, norm_size):
        super(SimpleRMSNormModel, self).__init__()
        self.rms_norm = RMSNorm(dim=norm_size)

    def forward(self, x):
        output = self.rms_norm(x)
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


class SimpleDistRMSNormModel(nn.Module):
    def __init__(self, norm_size):
        super(SimpleDistRMSNormModel, self).__init__()
        last_dim_split_comm_intf = TPYCollectiveComm()
        self.rms_norm = RMSNorm2D(norm_size, last_dim_split_comm_intf=last_dim_split_comm_intf)

    def forward(self, x):
        output = self.rms_norm(x)
        return output


class TestRMSNorm2dRsFirstDim(DistributedTest):
    world_size = 4
    diff_value = 1e-5

    @staticmethod
    def get_rms_norm_grad(dist_schedule, h, input_x, targets):
        # 生成一些随机输入数据和目标输出
        input_x_data = input_x.clone().detach()
        targets_data = targets.clone().detach()
        input_x_data.requires_grad_()
        # 创建模型实例
        if dist_schedule:
            model = SimpleDistRMSNormModel(h).npu()
        else:
            model = SimpleRMSNormModel(h).npu()

        # 创建损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        # 前向传播：计算预测值
        if dist_schedule:
            # s,b,E -> s/x,b,h/y
            input_x_data = auto_grad_scatter_along_first_dim(input_x_data, TPXCollectiveComm)
            input_x_data = auto_grad_scatter_along_last_dim(input_x_data, TPYCollectiveComm)
            outputs = model(input_x_data)
            outputs = auto_grad_sync_gather_along_first_dim(outputs, TPXCollectiveComm)
            outputs = auto_grad_sync_gather_along_last_dim(outputs, TPYCollectiveComm)
        else:
            outputs = model(input_x_data)

        # 计算损失
        loss = criterion(outputs, targets_data)
        # 反向传播：计算梯度
        loss.backward()
        return outputs, model.rms_norm.weight.grad

    def test_rms_norm_should_be_allclose_give_same_input_when_norm_1d_2d(self):
        set_random_seed(1)
        args = parse_args(None, True)
        set_args(args)
        tp, tp_x, tp_y = 4, 2, 2
        pp = self.world_size // tp
        h = 16
        seq = 8
        b = 1
        args.tp_2d = True
        args.tp_x = tp_x
        args.tp_y = tp_y
        # 生成一些随机输入数据和目标输出
        input_x = torch.randn(seq, b, h, requires_grad=True).npu()
        targets = torch.randn(seq, b, h).npu()
        destroy_model_parallel()
        initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_split_rank=None)
        # 2d rms_norm
        output_2d, weight_grad_2d = self.get_rms_norm_grad(dist_schedule=1, h=h, input_x=input_x, targets=targets)
        # 1d rms_norm
        output_1d, weight_grad_1d = self.get_rms_norm_grad(dist_schedule=0, h=h, input_x=input_x, targets=targets)
        assert torch.allclose(output_2d, output_1d, rtol=self.diff_value, atol=self.diff_value)
        weight_grad_1d = tensor_parallel.scatter_to_tensor_model_parallel_region(weight_grad_1d)
        weight_grad_2d = sync_reduce_scatter_along_first_dim(weight_grad_2d, TPXCollectiveComm)
        assert torch.allclose(weight_grad_2d, weight_grad_1d, rtol=self.diff_value, atol=self.diff_value)
