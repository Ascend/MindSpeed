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
    def gmm(a, b, batch_sizes, trans_b=False, gemm_fusion=False, original_weight=None):
        from mindspeed.ops.gmm import npu_gmm

        if trans_b:
            b = b.t()
        group_list = torch.cumsum(batch_sizes, dim=0).to('npu')
        return npu_gmm(a, b, bias=None, group_list=group_list, group_type=0, gemm_fusion=gemm_fusion, original_weight=original_weight)


def get_device_capability():
    return 9, 0
