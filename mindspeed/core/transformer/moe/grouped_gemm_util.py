# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

import torch


def grouped_gemm_is_available():
    try:
        from mindspeed.ops.gmm import npu_gmm
        return True
    except ImportError:
        return False


class Ops:
    @staticmethod
    def gmm(original_weight, a, b, batch_sizes, trans_b=False):
        from mindspeed.ops.gmm import npu_gmm

        if trans_b:
            b = b.t()
        group_list = torch.cumsum(batch_sizes, dim=0)
        return npu_gmm(original_weight, a, b, bias=None, group_list=group_list, group_type=0)


def get_device_capability():
    return 9, 0
