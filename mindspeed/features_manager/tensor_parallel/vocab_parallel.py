# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ReplaceIndexPutFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('vocab-parallel', optimization_level=2)

    def is_need_apply(self, args):
        """Check the feature is need to apply."""
        return self.optimization_level <= args.optimization_level

    def register_patches(self, patch_manager, args):
        from mindspeed.core.tensor_parallel.vocab_parallel.adaptor import mindspeed_vocab_parallel_embedding_forward
        patch_manager.register_patch('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                                     mindspeed_vocab_parallel_embedding_forward)
