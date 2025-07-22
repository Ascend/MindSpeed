# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class SwapAttentionFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('swap-attention', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--swap-attention', action='store_true', default=False,
                           help='switch to open swap-attention feature.'
                                'The default is False.')
        group.add_argument('--swap-modules', type=str, default="input_norm,self_attention,post_attention_norm",
                           help='Swap modules for model. Can be used together with "--swap-attention."')

    def validate_args(self, args):
        adaptive_recompute_device_size = getattr(args, 'adaptive-recompute-device-size', -1)
        adaptive_recompute_device_swap = getattr(args, 'adaptive-recompute-device-swap', False)
        if (adaptive_recompute_device_size > 0 or adaptive_recompute_device_swap) and args.swap_attention:
            raise AssertionError('adaptive selective recompute is not compatible with swap_attention feature')

        self.incompatible_check(args, 'adaptive_memory_optimization')
        is_enable_lora = hasattr(args, "lora_target_modules") and (len(args.lora_target_modules) != 0)
        if is_enable_lora:
            raise AssertionError('swap attention is not compatible with LoRA')

    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            from mindspeed.core.memory.swap_attention.adaptor import allowed_recomputing_swap_module_wrapper
            from megatron.legacy.model.transformer import ParallelTransformerLayer
            from megatron.core.transformer.transformer_layer import TransformerLayer
            from mindspeed.core.memory.common import transformer_block_checkpointed_forward
            if hasattr(args, "use_legacy_models") and not args.use_legacy_models:
                allowed_recomputing_swap_module_wrapper(TransformerLayer)
            else:
                allowed_recomputing_swap_module_wrapper(ParallelTransformerLayer)
            from mindspeed.core.memory.swap_attention.adaptor import setup_model_and_optimizer_wrapper
            patch_manager.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer_wrapper)

            from mindspeed.core.memory.common import linear_forward_main_grad_wrapper, linear_backward_main_grad_wrapper
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.forward',
                                linear_forward_main_grad_wrapper)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
                                linear_backward_main_grad_wrapper)
            patch_manager.register_patch(
                'megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward',
                transformer_block_checkpointed_forward)
