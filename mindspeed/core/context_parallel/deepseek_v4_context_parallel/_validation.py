# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.


def _resolve_deepseek_v4_seq_dim(layout, seq_dim, name):
    expected = 1 if layout == "BSND" else 0
    if seq_dim is None:
        return expected
    seq_dim = int(seq_dim)
    if seq_dim != expected:
        raise ValueError(f"{name} must be {expected} for layout {layout}, got {seq_dim}.")
    return seq_dim


def _get_deepseek_v4_batch_size(q, layout_q):
    if layout_q == "BSND":
        return int(q.shape[0])
    return 1
