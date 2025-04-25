# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from mindspeed.features_manager.feature import MindSpeedFeature


class RecomputeNormFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('recompute-norm')

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--recompute-norm', action='store_true',
                            help='Recompute norm in Transformer Layers')
        group.add_argument('--recompute-norm-num-layers', type=int, default=None,
                            help='Recompute norm num layers, can be used together with activation function recompute. ')

    def validate_args(self, args):
        if args.recompute_norm_num_layers is not None:
            if not isinstance(args.recompute_norm_num_layers, int):
                raise TypeError('--recompute-norm-num-layers must be an integer.')
            if args.recompute_norm_num_layers < 0:
                raise AssertionError('--recompute-norm-num-layers cannot be less than 0.')
            if args.recompute_norm_num_layers > args.num_layers:
                raise ValueError(f'--recompute-norm-num-layers ({args.recompute_norm_num_layers}) '
                                            f'cannot be greater than --num-layers ({args.num_layers}).')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.memory.recompute.norm.adaptor import mindspeed_norm_recompute_forward, build_norm_recompute_layer_wrapper

        if getattr(args, self.feature_name, None):
            patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer.forward', mindspeed_norm_recompute_forward)
            patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock._build_layers', build_norm_recompute_layer_wrapper)
    