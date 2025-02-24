from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class LlavaModel(MindSpeedFeature):

    def __init__(self):
        super().__init__('llava-multimodal', default_patches=True)

    def register_patches(self, patch_manager, args):
        from mindspeed.core.models.multimodal.llava_model import llava_init_wrapper
        patch_manager.register_patch('megatron.core.models.multimodal.llava_model.LLaVAModel.__init__', llava_init_wrapper)
