from typing import List

from mindspeed.features_manager import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class MindSpeedFeaturesManager:
    FEATURES_LIST = []

    @classmethod
    def set_features_list(cls, features_list: List[MindSpeedFeature]):
        """Set features list"""
        cls.FEATURES_LIST[:] = features_list

    @classmethod
    def apply_features_pre_patches(cls, mindspeed_args):
        """Apply pre patches of all features."""
        for feature in cls.FEATURES_LIST:
            if feature.is_need_apply(mindspeed_args):
                feature.pre_register_patches(MindSpeedPatchesManager, mindspeed_args)
        MindSpeedPatchesManager.apply_patches()

    @classmethod
    def apply_features_patches(cls, mindspeed_args):
        """Apply patches of all features."""
        for feature in cls.FEATURES_LIST:
            if feature.is_need_apply(mindspeed_args):
                feature.register_patches(MindSpeedPatchesManager, mindspeed_args)
        MindSpeedPatchesManager.apply_patches()

    @classmethod
    def register_features_args(cls, parser):
        """Parse arguments of all features."""
        for feature in cls.FEATURES_LIST:
            feature.register_args(parser)

    @classmethod
    def pre_validate_features_args(cls, args):
        """Pre-validate arguments of all features. Used to bypass megatron arguments validation.
        Example:
            pre_validate_features_args(args)  # old_x = args.x; args.x = new_x
            args = validate_args(args, defaults)  # bypass args.x validation
            post_validate_features_args(args=args)  # args.x = old_x
        """
        for feature in cls.FEATURES_LIST:
            feature.pre_validate_args(args)

    @classmethod
    def post_validate_features_args(cls, args):
        """Post-validate arguments of all features. Used to bypass megatron arguments validation.
        Example:
            pre_validate_features_args(args)  # old_x = args.x; args.x = new_x
            args = validate_args(args, defaults)  # bypass args.x validation
            post_validate_features_args(args=args)  # args.x = old_x
        """
        for feature in cls.FEATURES_LIST:
            feature.post_validate_args(args)

    @classmethod
    def validate_features_args(cls, args):
        """Validate arguments of all features."""
        for feature in cls.FEATURES_LIST:
            feature.validate_args(args)
