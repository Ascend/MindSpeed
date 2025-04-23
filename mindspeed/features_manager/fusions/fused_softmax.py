from mindspeed.features_manager.feature import MindSpeedFeature


class FusedSoftmaxFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('fused-softmax', optimization_level=0)

    def register_patches(self, pm, args):
        from mindspeed.core.fusions.fused_softmax import is_kernel_available, ScaledUpperTriangMaskedSoftmax, \
            ScaledMaskedSoftmax, ScaledSoftmax, forward_fused_softmax
        pm.register_patch('megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax',
                          ScaledUpperTriangMaskedSoftmax)
        pm.register_patch('megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax', ScaledMaskedSoftmax)
        pm.register_patch('megatron.core.fusions.fused_softmax.ScaledSoftmax', ScaledSoftmax)
        pm.register_patch('megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available',
                          is_kernel_available)
        pm.register_patch('megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax',
                          forward_fused_softmax)
