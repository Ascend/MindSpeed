# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from mindspore import ops


class _RingAttentionProxy:
    def npu_ring_attention_update(self, *args, **kwargs):
        return ops.ring_attention_update(*args, **kwargs)


_RING_PROXY = _RingAttentionProxy()


def _ring_atten_patched_load(*_args, **_kwargs):
    return _RING_PROXY