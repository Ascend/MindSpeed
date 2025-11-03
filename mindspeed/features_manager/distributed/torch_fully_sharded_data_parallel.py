"""Define TorchFullyShardedDataParallel feature.

Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from argparse import ArgumentParser, Namespace
from logging import getLogger

from mindspeed.features_manager.feature import MindSpeedFeature

LOG = getLogger(__name__)


class TorchFullyShardedDataParallelFeature(MindSpeedFeature):
    """Torch Fully Sharded Data Parallel feature."""

    def __init__(self):
        super().__init__(feature_name='use-torch-fsdp2', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--fsdp2-config-path',
                           type=str)

    def validate_args(self, args):
        if not getattr(args, 'use_torch_fsdp2', False) and getattr(args, 'fsdp2_config_path', None):
            raise AssertionError('--fsdp2-config-path only used when --use-torch-fsdp2')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.distributed.torch_fully_sharded_data_parallel.torch_fully_sharded_data_parallel_adaptor import torch_fully_sharded_data_parallel_init
        patch_manager.register_patch('megatron.core.distributed.TorchFullyShardedDataParallel.__init__',
                                torch_fully_sharded_data_parallel_init)

        # configuration patches
        from mindspeed.core.distributed.torch_fully_sharded_data_parallel.distributed_data_parallel_config_adaptor import distributed_data_parallel_init_wrapper
        patch_manager.register_patch('megatron.core.distributed.DistributedDataParallelConfig.__init__', distributed_data_parallel_init_wrapper)

        # PATCH: Fix Megatron checkpoint loading compatibility for torch_dcp format
        from mindspeed.checkpointing import load_checkpoint
        patch_manager.register_patch('megatron.training.checkpointing.load_checkpoint', load_checkpoint)

        # BUGFIX: Fix Megatron Meta Initialization
        from mindspeed.core.distributed.torch_fully_sharded_data_parallel.training import get_model
        patch_manager.register_patch('megatron.training.training.get_model', get_model)

        # BUGFIX: Correct get_data_parallel_group_if_dtensor for 2D device mesh in Megatron
        from mindspeed.core.distributed.torch_fully_sharded_data_parallel.utils import get_data_parallel_group_if_dtensor
        patch_manager.register_patch('megatron.core.utils.get_data_parallel_group_if_dtensor', get_data_parallel_group_if_dtensor)

        # Patch distributed backend to support multiple backends.
        from mindspeed.core.distributed.torch_fully_sharded_data_parallel.arguments import _add_distributed_args_wrapper
        patch_manager.register_patch('megatron.training.arguments._add_distributed_args', _add_distributed_args_wrapper)