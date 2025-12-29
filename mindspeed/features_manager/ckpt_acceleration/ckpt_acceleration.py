from argparse import Namespace
from logging import getLogger

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager

LOG = getLogger(__name__)


class CkptAccelerationFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('ckpt-acceleration', 0)

    def is_need_apply(self, args):
        LOG.info(f"ckpt acceleration is_need_apply patches enter"
              f"ckpt_acceleration: {getattr(args, 'ckpt_acceleration', False)}")
        if not hasattr(args, 'ckpt_acceleration') or not args.ckpt_acceleration:
            return False
        return True

    def pre_register_patches(self, patch_manager: MindSpeedPatchesManager, args: Namespace):
        if not hasattr(args, 'ckpt_acceleration') or not args.ckpt_acceleration:
            return
        from mindspeed.core.dist_checkpointing.checkpoint_adaptor import (save_wrapper, validate_global_plan_wrapper,
                                                                          validate_non_overlapping_shards_metadata_wrapper)
        patch_manager.register_patch('megatron.core.dist_checkpointing.save', save_wrapper)
        patch_manager.register_patch('torch.distributed.checkpoint.default_planner._validate_global_plan',
                                     validate_global_plan_wrapper)
        patch_manager.register_patch('torch.distributed._shard.sharding_spec._internals.' +
                                     'validate_non_overlapping_shards_metadata',
                                     validate_non_overlapping_shards_metadata_wrapper)
        LOG.info(f"register ckpt acceleration patches success")