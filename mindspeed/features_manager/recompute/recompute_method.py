# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from mindspeed.features_manager.feature import MindSpeedFeature


class RecomputeMethodFeature(MindSpeedFeature):
    """
    Additional setting for recompute_feature.
    """
    def __init__(self):
        super().__init__('recompute-method', optimization_level=2)

    def register_patches(self, patch_manager, args):
        self.register_basic_patches(patch_manager, args)

    def register_basic_patches(self, patch_manager, args):
        if (getattr(args, 'recompute_method', False) and args.recompute_method == 'block') and not getattr(args, 'swap_attention', False):
            from mindspeed.core.memory.common import transformer_block_checkpointed_forward
            patch_manager.register_patch(
                'megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward',
                transformer_block_checkpointed_forward)
