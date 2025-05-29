# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

from copy import deepcopy

import torch

from tests_extend.unit_tests.common import DistributedTest

from mindspeed.multi_modal.conv3d.conv3d_depth_parallel import Conv3DSequenceParallel

torch.manual_seed(1234)


class TestConv3dSequenceParallel(DistributedTest):
    world_size = 4

    def test_conv3d_depth_parallel(self):
        group_ranks = list(range(self.world_size))
        group = torch.distributed.new_group(group_ranks)

        conv_without_sp = Conv3DSequenceParallel(pg=group, in_channels=8, out_channels=4, kernel_size=(3, 3, 3),
                                                 stride=(4, 4, 4), param_async=True, sp_size=1)

        conv_with_sp = Conv3DSequenceParallel(pg=group, in_channels=8, out_channels=4, kernel_size=(3, 3, 3),
                                              stride=(4, 4, 4), param_async=True, sp_size=4)

        conv_with_sp.conv3d = deepcopy(conv_without_sp.conv3d)
        input_data = torch.randn(1, 8, 128, 128, 128).npu().to(torch.bfloat16)
        output_without_sp = conv_without_sp(input_data)
        output_with_sp = conv_with_sp(input_data)

        assert torch.allclose(output_without_sp, output_with_sp, rtol=0.001, atol=0.001)
