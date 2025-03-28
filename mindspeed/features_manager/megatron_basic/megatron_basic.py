from mindspeed.features_manager.feature import MindSpeedFeature


class MegatronBasicFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('megatron-basic', optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--use-fused-rmsnorm", action='store_true', help="Use fused rmsnorm.")

    def register_patches(self, patch_manager, args):
        self.register_basic_patches(patch_manager, args)

    def register_basic_patches(self, pm, args):
        # args parser patch
        from mindspeed.core.megatron_basic.arguments_basic import parse_args_wrapper, validate_args_wrapper, print_args_wrapper, transformer_config_post_init_wrapper
        pm.register_patch('megatron.training.arguments.parse_args', parse_args_wrapper)
        pm.register_patch('megatron.training.arguments.validate_args', validate_args_wrapper)
        pm.register_patch('megatron.training.arguments._print_args', print_args_wrapper)
        pm.register_patch('megatron.training.yaml_arguments.validate_yaml', validate_args_wrapper)
        pm.register_patch('megatron.training.yaml_arguments._print_args', print_args_wrapper)
        pm.register_patch("megatron.core.transformer.transformer_config.TransformerConfig.__post_init__", transformer_config_post_init_wrapper)

        # initialization patches
        from mindspeed.core.megatron_basic.megatron_basic import _set_cuda_rng_state, _compile_dependencies
        pm.register_patch('megatron.core.tensor_parallel.random._set_cuda_rng_state', _set_cuda_rng_state)
        pm.register_patch('megatron.training.initialize._compile_dependencies', _compile_dependencies)

        # norm patches
        from mindspeed.core.megatron_basic.megatron_basic import PTNorm
        pm.register_patch('megatron.core.models.gpt.gpt_layer_specs.LNImpl', PTNorm)
        pm.register_patch('megatron.core.transformer.torch_norm.WrappedTorchNorm', PTNorm)
        pm.register_patch('megatron.core.transformer.transformer_block.LayerNormImpl', PTNorm)
        pm.register_patch('megatron.core.extensions.transformer_engine.TENorm', PTNorm)

        # coalescing_manager patches
        from mindspeed.core.distributed.param_and_grad_buffer import start_param_sync, finish_param_sync, start_grad_sync, finish_grad_sync
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_param_sync', start_param_sync)
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.finish_param_sync', finish_param_sync)
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_grad_sync', start_grad_sync)
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.finish_grad_sync', finish_grad_sync)


