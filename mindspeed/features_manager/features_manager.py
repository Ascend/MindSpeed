from megatron_adaptor.features_manager.features_manager import FeaturesManager
from mindspeed.patch_utils import MindSpeedPatchesManager


class MindSpeedFeaturesManager(FeaturesManager):
    """Feature manager that keeps MindSpeed patch ownership separate from MA."""

    FEATURES_LIST = []

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
    def remove_patches(cls):
        MindSpeedPatchesManager.remove_patches()
