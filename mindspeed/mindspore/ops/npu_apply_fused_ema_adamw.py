# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import torch_npu


class _FusedEmaAdamwProxy:
    def npu_apply_fused_ema_adamw(self, *args, **kwargs):
        return torch_npu.npu_apply_fused_ema_adamw(*args, **kwargs)


_Fused_PROXY = _FusedEmaAdamwProxy()


def _fused_ema_adamw_patched_load(*_args, **_kwargs):
    return _Fused_PROXY
