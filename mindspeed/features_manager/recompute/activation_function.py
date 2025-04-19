from mindspeed.features_manager.feature import MindSpeedFeature


class RecomputeActivationFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('recompute-activation-function')

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--recompute-activation-function', action='store_true',
                           help='Recompute the activation function in MLP layers.')
        group.add_argument('--recompute-activation-function-num-layers', type=int, default=None,
                           help='Can be used together with "--recompute-method block." '
                           'and "--recompute-num-layers". ')

    def validate_args(self, args):
        if args.recompute_activation_function_num_layers is not None:
            if not isinstance(args.recompute_activation_function_num_layers, int):
                raise TypeError('--recompute-activation-function-num-layers must be an integer.')
            if args.recompute_activation_function_num_layers < 0:
                raise AssertionError('--recompute-activation-function-num-layers cannot be less than 0.')
            if args.recompute_activation_function_num_layers > args.num_layers:
                raise ValueError(f'--recompute-activation-function-num-layers ({args.recompute_activation_function_num_layers}) '
                                            f'cannot be greater than --num-layers ({args.num_layers}).')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.memory.recompute.activation.adaptor import mindspeed_activation_recompute_forward
        from mindspeed.core.transformer.transformer import parallel_transformer_layer_init_wrapper

        if getattr(args, self.feature_name, None):
            patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer.__init__',
                                          parallel_transformer_layer_init_wrapper)
            patch_manager.register_patch('megatron.core.transformer.mlp.MLP.forward', mindspeed_activation_recompute_forward)
