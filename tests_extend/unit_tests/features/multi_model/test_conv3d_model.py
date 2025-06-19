# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from copy import deepcopy
import torch
from torch import nn, optim

from tests_extend.unit_tests.common import DistributedTest

from mindspeed.multi_modal.conv3d.conv3d_depth_parallel import Conv3DSequenceParallel

torch.manual_seed(1234)


class TestConv3dModel(nn.Module):
    def __init__(self, conv1, conv2, conv3):
        super(TestConv3dModel, self).__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.handles = []

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def wait_all_handles(self):
        self.handles = []
        for module in [self.conv1, self.conv2, self.conv3]:
            if isinstance(module, Conv3DSequenceParallel):
                module.wait_param_grad_reduce_handles()


class TestConv3dDepthParallel(DistributedTest):
    world_size = 4

    def test_conv3d_depth_parallel(self):
        group_ranks = list(range(self.world_size))
        group = torch.distributed.new_group(group_ranks)

        conv1 = Conv3DSequenceParallel(pg=group, in_channels=8, out_channels=4, kernel_size=(3, 3, 3),
                                       stride=(4, 4, 4), param_async=True, sp_size=1, dtype=torch.bfloat16)
        conv2 = Conv3DSequenceParallel(pg=group, in_channels=4, out_channels=2, kernel_size=(3, 3, 3),
                                       stride=(4, 4, 4), param_async=True, sp_size=1, dtype=torch.bfloat16)
        conv3 = Conv3DSequenceParallel(pg=group, in_channels=2, out_channels=1, kernel_size=(3, 3, 3),
                                       stride=(4, 4, 4), param_async=True, sp_size=1, dtype=torch.bfloat16)

        conv1_clone = Conv3DSequenceParallel(pg=group, in_channels=8, out_channels=4, kernel_size=(3, 3, 3),
                                             stride=(4, 4, 4), param_async=True, sp_size=4, dtype=torch.bfloat16)
        conv2_clone = Conv3DSequenceParallel(pg=group, in_channels=4, out_channels=2, kernel_size=(3, 3, 3),
                                             stride=(4, 4, 4), param_async=True, sp_size=4, dtype=torch.bfloat16)
        conv3_clone = Conv3DSequenceParallel(pg=group, in_channels=2, out_channels=1, kernel_size=(3, 3, 3),
                                             stride=(4, 4, 4), param_async=True, sp_size=4, dtype=torch.bfloat16)

        conv1_clone.conv3d = deepcopy(conv1.conv3d)
        conv2_clone.conv3d = deepcopy(conv2.conv3d)
        conv3_clone.conv3d = deepcopy(conv3.conv3d)

        model1 = TestConv3dModel(conv1, conv2, conv3)
        model2 = TestConv3dModel(conv1_clone, conv2_clone, conv3_clone)

        criterion = nn.MSELoss()
        optimizer1 = optim.Adam(model1.parameters(), lr=1e-5)
        optimizer2 = optim.Adam(model2.parameters(), lr=1e-5)

        input_data = torch.randn(1, 8, 128, 128, 128).npu().to(torch.bfloat16)
        input_data2 = deepcopy(input_data)
        target = [torch.ones(1, 1, 2, 2, 2).npu().to(torch.float32) for _ in range(2000)]

        loss1_lst = []
        loss2_lst = []
        epoch = 2000
        for i in range(epoch):
            optimizer1.zero_grad()
            input_data.requires_grad = True
            output_data1 = model1(input_data)
            loss1 = criterion(output_data1.float(), target[i].float())
            loss1.backward()
            optimizer1.step()
            loss1_lst.append(float(loss1))
        torch.distributed.barrier()

        for i in range(epoch):
            optimizer2.zero_grad()
            input_data2.requires_grad = True
            output_data2 = model2(input_data2)
            loss2 = criterion(output_data2.float(), target[i].float())
            loss2.backward()
            model2.wait_all_handles()
            optimizer2.step()
            loss2_lst.append(float(loss2))

        if torch.distributed.get_rank() == 0:
            with open('conv3d_without_sp.log', 'a') as f:
                for item in loss1_lst:
                    f.write('loss: ' + str(item) + '\n')

            with open('conv3d_with_sp.log', 'a') as f:
                for item in loss2_lst:
                    f.write('loss: ' + str(item) + '\n')

        absolute_errors = [abs(a - b) for a, b in zip(loss1_lst, loss2_lst)]
        total_error = sum(absolute_errors)
        average_error = total_error / len(loss1_lst)
        print(f"average error: {average_error}")
