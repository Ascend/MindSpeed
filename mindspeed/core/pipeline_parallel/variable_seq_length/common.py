"""Define common class for variable length feature.

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""

from typing import Protocol

import torch


class Config(Protocol):
    """Define a protocol for configuration."""

    pipeline_dtype: torch.dtype
    """dtype used in p2p communication, usually params_dtype"""

    batch_p2p_comm: bool
    """Use batch_isend_irecv instead of individual isend/irecv calls.
    Must be False if overlap_p2p_comm is True.
    """

    batch_p2p_sync: bool
    """When using batch_isend_irecv,
    do a cuda.device.synchronize afterward to work around a bug in
    older version of PyTorch.
    """

    use_ring_exchange_p2p: bool
    """Use custom ring_exchange kernel
    instead of torch.distributed.batch_isend_irecv(). Requires
    custom built torch with torch.distributed.ring_exchange.
    """

    variable_seq_lengths: bool
    """Support for variable sequence lengths across microbatches.
    Setting this communicates the size of tensors during pipeline parallelism
    communication, because of this extra overhead it
    should only be set if the sequence length varies
    by microbatch within a global batch.
    """
