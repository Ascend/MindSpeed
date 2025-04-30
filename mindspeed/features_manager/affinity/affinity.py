# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from mindspeed.features_manager.feature import MindSpeedFeature


class AffinityFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('affinity', optimization_level=1)

    def is_need_apply(self, args):
        """Check the feature is need to apply."""
        return self.optimization_level <= args.optimization_level

    def register_patches(self, patch_manager, args):
        from mindspeed.core.tensor_parallel.cross_entropy import calculate_predicted_logits
        # use logical negation followed by multiplication to achieve the same effect as setting selected elements to zero
        patch_manager.register_patch(
            'megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.calculate_predicted_logits',
            calculate_predicted_logits)
