from mindspeed.features_manager.feature import MindSpeedFeature


class FusedSwigluFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('use-swiglu', optimization_level=0)

    def register_patches(self, patch_manager, args):
        from mindspeed.core.fusions.fused_bias_swiglu import SwiGLUFunction, BiasSwiGLUFunction
        patch_manager.register_patch('megatron.core.fusions.fused_bias_swiglu.SwiGLUFunction', SwiGLUFunction)
        patch_manager.register_patch('megatron.core.fusions.fused_bias_swiglu.BiasSwiGLUFunction', BiasSwiGLUFunction)
