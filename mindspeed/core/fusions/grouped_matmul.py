# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch


class Ops:
    @staticmethod
    def gmm(a, b, batch_sizes, trans_b=False, gemm_fusion=False, original_weight=None):
        from mindspeed.ops.gmm import npu_gmm

        if trans_b:
            b = b.t()
        group_list = torch.cumsum(batch_sizes, dim=0).to('npu')
        return npu_gmm(
            a,
            b,
            bias=None,
            group_list=group_list,
            group_type=0,
            gemm_fusion=gemm_fusion,
            original_weight=original_weight,
        )

    @staticmethod
    def gmm_w4a8(a, b, batch_sizes, trans_b=False, original_weight=None, block_size=32):
        from mindspeed.ops.gmm_w4a8 import npu_gmm

        if trans_b:
            b = b.t()
        group_list = torch.cumsum(batch_sizes, dim=0).to('npu')
        return npu_gmm(
            a, b, bias=None, group_list=group_list, group_type=0, original_weight=original_weight, block_size=block_size
        )


def grouped_gemm_is_available():
    try:
        from mindspeed.ops.gmm import npu_gmm  # noqa: F401

        return True
    except ImportError:
        return False


def assert_grouped_gemm_is_available():
    if not grouped_gemm_is_available():
        raise ImportError("from mindspeed.ops.gmm import npu_gmm failed.")


def get_device_capability(device=None):
    return 9, 0
