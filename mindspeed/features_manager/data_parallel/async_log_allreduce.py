# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0 OR MIT

from argparse import ArgumentParser, Namespace
from logging import getLogger

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager

LOG = getLogger(__name__)


class AsyncLogAllreduceFeature(MindSpeedFeature):
    """Layer Zero feature."""

    def __init__(
        self,
        feature_name: str = "async-log-allreduce",
        optimization_level: int = 2,
    ):
        super().__init__(feature_name, optimization_level)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='overlap_p2p_comm_or_async_log_allreduce_')
        group.add_argument(
            '--async-log-allreduce',
            action='store_true',
            help='Transform the AllReduce operation used for transmitting log information into an asynchronous operation to reduce communication overhead. This is useful in cross-DataCenter (DC) training.')

    def register_patches(
        self,
        patch_manager: MindSpeedPatchesManager,
        args: Namespace,
    ):
        # pylint: disable=import-outside-toplevel
        from mindspeed.core.data_parallel.async_log_allreduce import train_step
        patch_manager.register_patch('megatron.training.training.train_step', train_step)
