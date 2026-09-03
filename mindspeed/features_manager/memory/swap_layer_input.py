# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class SwapLayerInputFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('swap-layer-input', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            '--swap-layer-input',
            action='store_true',
            default=False,
            help='switch to open swap layer input feature. The default is False.',
        )

    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            from mindspeed.core.memory.swap_layer_input.swap_layer_input import (
                swap_layer_input_init_wrapper,
                swap_layer_input_forward_wrapper,
                swap_layer_input_fboverlap_forward_wrapper,
                swap_layer_input_fboverlap_1f1b_wrapper,
                swap_layer_input_fboverlap_backward_wrapper,
            )

            patch_manager.register_patch(
                'megatron.core.transformer.transformer_layer.TransformerLayer.__init__', swap_layer_input_init_wrapper
            )
            patch_manager.register_patch(
                'megatron.core.transformer.transformer_layer.TransformerLayer.forward', swap_layer_input_forward_wrapper
            )
            if getattr(args, 'moe_fb_overlap', None):
                patch_manager.register_patch(
                    'mindspeed.core.transformer.moe.moe_feature.fb_overlap.overlap_funcs.fwd.transformer_layer_forward_moe',
                    swap_layer_input_fboverlap_forward_wrapper,
                )
                patch_manager.register_patch(
                    'mindspeed.core.transformer.moe.moe_feature.fb_overlap.overlap_funcs.fwd.transformer_layer_forward_dense',
                    swap_layer_input_fboverlap_forward_wrapper,
                )
                patch_manager.register_patch(
                    'mindspeed.core.transformer.moe.moe_feature.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_dense_backward_dense_overlaping',
                    swap_layer_input_fboverlap_1f1b_wrapper,
                )
                patch_manager.register_patch(
                    'mindspeed.core.transformer.moe.moe_feature.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_moe_backward_dense_overlaping',
                    swap_layer_input_fboverlap_1f1b_wrapper,
                )
                patch_manager.register_patch(
                    'mindspeed.core.transformer.moe.moe_feature.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_dense_backward_moe_overlaping',
                    swap_layer_input_fboverlap_1f1b_wrapper,
                )
                patch_manager.register_patch(
                    'mindspeed.core.transformer.moe.moe_feature.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_moe_backward_moe_overlaping',
                    swap_layer_input_fboverlap_1f1b_wrapper,
                )
                patch_manager.register_patch(
                    'mindspeed.core.transformer.moe.moe_feature.fb_overlap.transformer_layer.transformer_layer_backward',
                    swap_layer_input_fboverlap_backward_wrapper,
                )
