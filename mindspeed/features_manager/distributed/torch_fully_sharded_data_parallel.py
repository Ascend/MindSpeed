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
        group.add_argument('--fsdp2-config-str',
                           type=str,
                           help='fsdp2 fully_shard interface parameters str, ex: \'key1=value1;key2=value2\''
                                'note: for parameters of type class, we will split them into the base type to input the key.'
                                'for example, for the parameter mp_policy(MixedPrecisionPolicy), we will name its sub-attribute as'
                                'mp_policy_param_dtype and input it as the key.Currently, parameters such as module, offload_policy, shard_placement_fn'
                                'and mesh are not supported for input')

    def validate_args(self, args):
        if not args.use_torch_fsdp2 and args.fsdp2_config_str:
            raise AssertionError('--fsdp2-config-str only used when --use-torch-fsdp2')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.distributed.torch_fully_sharded_data_parallel.torch_fully_sharded_data_parallel_adaptor import torch_fully_sharded_data_parallel_init
        patch_manager.register_patch('megatron.core.distributed.TorchFullyShardedDataParallel.__init__',
                                torch_fully_sharded_data_parallel_init)

        # configuration patches
        from mindspeed.core.distributed.torch_fully_sharded_data_parallel.distributed_data_parallel_config_adaptor import distributed_data_parallel_init_wrapper
        patch_manager.register_patch('megatron.core.distributed.DistributedDataParallelConfig.__init__', distributed_data_parallel_init_wrapper)