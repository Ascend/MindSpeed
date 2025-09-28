import time
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ResetBucketGroupOrderFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('reset-bucket-group-order', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--reset-bucket-group-order',
                           action='store_true', default=False,
                           help='If true, forward compute with right overlap param all-gather order.')

    def validate_args(self, args):
        reset_bucket_group_order = getattr(args, "reset_bucket_group_order", False)
        overlap_param_gather = getattr(args, "overlap_param_gather", False)
        if reset_bucket_group_order and not overlap_param_gather:
            raise AssertionError('overlap param gather is compatible with reset bucket group order')

    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            reset_bucket_group_order = getattr(args, "reset_bucket_group_order", False)
            if reset_bucket_group_order:
                from mindspeed.core.distributed.reset_bucket_group_order_feature.distributed_data_parallel_config import \
                    distributed_data_parallel_config_init_wrapper
                patch_manager.register_patch(
                    'megatron.core.distributed.distributed_data_parallel_config.DistributedDataParallelConfig.__init__',
                    distributed_data_parallel_config_init_wrapper)

                from mindspeed.core.distributed.reset_bucket_group_order_feature.distributed_data_parallel import \
                    distributed_data_parallel_init_wrapper
                patch_manager.register_patch(
                    'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.DistributedDataParallel.__init__',
                    distributed_data_parallel_init_wrapper)

                from mindspeed.core.distributed.reset_bucket_group_order_feature.distributed_data_parallel import _make_forward_pre_hook
                patch_manager.register_patch(
                    'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_forward_pre_hook',
                    _make_forward_pre_hook)