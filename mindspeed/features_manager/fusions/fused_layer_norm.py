from mindspeed.features_manager.feature import MindSpeedFeature


class FusedLayerNormFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('fused-layernorm', optimization_level=0)

    def register_patches(self, pm, args):
        from mindspeed.core.fusions.fused_layer_norm import FusedLayerNormAffineFunction, FastLayerNormFN
        pm.register_patch('megatron.core.fusions.fused_layer_norm.FusedLayerNormAffineFunction',
                          FusedLayerNormAffineFunction)
        pm.register_patch('megatron.core.fusions.fused_layer_norm.FastLayerNormFN', FastLayerNormFN)
