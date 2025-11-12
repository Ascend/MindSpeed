from argparse import ArgumentParser
import torch

from mindspeed.features_manager.feature import MindSpeedFeature


_GRAD_TOKENS = {'fp32', 'fp16', 'bf16', 'fp8'}
_MAIN_PARAM_TOKENS = {'fp32', 'fp16'}
_EXP_AVG_TOKENS = {'fp32', 'fp16', 'bf16', 'fp8', 'mxfp8', 'fp8_e4m3', 'hif8_15'}
_EXP_AVG_SQ_TOKENS = {'fp32', 'fp16', 'bf16', 'fp8', 'mxfp8', 'fp8_e4m3', 'fp8_e5m2', 'hif8_224'}
_PAIRING_RULES = {
    'fp8': 'fp8',
    'mxfp8': 'mxfp8',
    'fp8_e4m3': 'fp8_e4m3',
    'hif8_15': 'hif8_224',
}
_TOKEN_TO_DTYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp8': torch.uint8,
    'mxfp8': torch.float32,
    'fp8_e4m3': torch.float32,
    'fp8_e5m2': torch.float32,
    'hif8_15': torch.float32,
    'hif8_224': torch.float32,
}
_LOW_PRECISION_STATES = set(_PAIRING_RULES.keys()) | set(_PAIRING_RULES.values())
_REVERSE_PAIRINGS = {value: key for key, value in _PAIRING_RULES.items()}
_QUANT_STATE_CHOICES = ('fp8', 'hif8', 'mxfp8', 'fp16')


def _dtype_to_token(value, default='fp32'):
    reverse_map = {
        torch.float32: 'fp32',
        torch.float16: 'fp16',
        torch.bfloat16: 'bf16',
        torch.uint8: 'fp8',
    }
    if isinstance(value, torch.dtype):
        return reverse_map.get(value, default)
    if isinstance(value, str):
        return value.lower()
    return default


def _token_to_dtype(token: str) -> torch.dtype:
    return _TOKEN_TO_DTYPE.get(token.lower(), torch.float32)


class LowPrecisionOptimizerFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('low-precision-optimizer', optimization_level=0)

    @staticmethod
    def _normalize_flags(args):
        tokens = {
            'main_grads_dtype': _dtype_to_token(getattr(args, 'main_grads_dtype', 'fp32')),
            'main_params_dtype': _dtype_to_token(getattr(args, 'main_params_dtype', 'fp32')),
            'exp_avg_dtype': _dtype_to_token(getattr(args, 'exp_avg_dtype', 'fp32')),
            'exp_avg_sq_dtype': _dtype_to_token(getattr(args, 'exp_avg_sq_dtype', 'fp32')),
        }

        quant_states = getattr(args, 'quant_states', None)
        if isinstance(quant_states, str):
            quant_states = quant_states.lower()
        if quant_states is not None and quant_states not in _QUANT_STATE_CHOICES:
            raise AssertionError(
                f"Low precision optimizer only supports quant_states {_QUANT_STATE_CHOICES}, got '{quant_states}'."
            )

        use_quant = bool(quant_states) or bool(getattr(args, 'quant_grads', False))
        requested_precision = bool(getattr(args, 'use_precision_aware_optimizer', False))
        requires_precision = any(token != 'fp32' for token in tokens.values())

        if use_quant and (requested_precision or requires_precision):
            raise AssertionError(
                'Precision-aware optimizer dtype overrides cannot be combined with quant optimizer.'
            )

        if use_quant:
            for attr, token in tokens.items():
                setattr(args, attr, _token_to_dtype(token))
            args.use_precision_aware_optimizer = False
            args.use_quant_optimizer = True

            main_grads_token = tokens['main_grads_dtype']
            quant_grads_requested = bool(getattr(args, 'quant_grads', False))
            if main_grads_token in {'fp16', 'bf16'}:
                args.quant_grads = True
                args.quant_grads_dtype = main_grads_token
            elif quant_grads_requested:
                quant_grad_dtype = getattr(args, 'quant_grads_dtype', None)
                if quant_grad_dtype in {'fp16', 'bf16'}:
                    args.quant_grads_dtype = quant_grad_dtype
                else:
                    args.quant_grads_dtype = 'fp16'
            else:
                args.quant_grads_dtype = None

            args.quant_states = quant_states
            return False, True

        args.quant_states = quant_states
        if getattr(args, 'quant_grads', False):
            args.quant_grads = False
        args.quant_grads_dtype = None

        if requested_precision or requires_precision:
            for attr, token in tokens.items():
                setattr(args, attr, token)
            args.use_precision_aware_optimizer = True
            args.use_quant_optimizer = False
            return True, False

        for attr, token in tokens.items():
            setattr(args, attr, _token_to_dtype(token))
        args.use_precision_aware_optimizer = False
        args.use_quant_optimizer = False
        return False, False

    @staticmethod
    def _validate_tokens(args):
        if not getattr(args, 'use_precision_aware_optimizer', False):
            return

        main_grads_token = _dtype_to_token(args.main_grads_dtype)
        main_params_token = _dtype_to_token(args.main_params_dtype)
        exp_avg_token = _dtype_to_token(args.exp_avg_dtype)
        exp_avg_sq_token = _dtype_to_token(args.exp_avg_sq_dtype)

        args.main_grads_dtype = main_grads_token
        args.main_params_dtype = main_params_token
        args.exp_avg_dtype = exp_avg_token
        args.exp_avg_sq_dtype = exp_avg_sq_token

        if main_grads_token not in _GRAD_TOKENS:
            raise AssertionError(f"Unsupported main_grads_dtype: {main_grads_token}")
        if main_params_token not in _MAIN_PARAM_TOKENS:
            raise AssertionError(f"Unsupported main_params_dtype: {main_params_token}")
        if exp_avg_token not in _EXP_AVG_TOKENS:
            raise AssertionError(f"Unsupported exp_avg_dtype: {exp_avg_token}")
        if exp_avg_sq_token not in _EXP_AVG_SQ_TOKENS:
            raise AssertionError(f"Unsupported exp_avg_sq_dtype: {exp_avg_sq_token}")

        low_precision_avg = exp_avg_token in _LOW_PRECISION_STATES
        low_precision_sq = exp_avg_sq_token in _LOW_PRECISION_STATES

        if low_precision_avg and low_precision_sq:
            expected_sq = _PAIRING_RULES.get(exp_avg_token)
            if expected_sq is not None and exp_avg_sq_token != expected_sq:
                raise AssertionError(
                    f"exp_avg_dtype '{exp_avg_token}' requires exp_avg_sq_dtype '{expected_sq}',"
                    f" but got '{exp_avg_sq_token}'."
                )
            expected_avg = _REVERSE_PAIRINGS.get(exp_avg_sq_token)
            if expected_avg is not None and exp_avg_token != expected_avg:
                raise AssertionError(
                    f"exp_avg_sq_dtype '{exp_avg_sq_token}' requires exp_avg_dtype '{expected_avg}'."
                )

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        option_strings = {opt for action in parser._actions for opt in action.option_strings}
        if '--use-precision-aware-optimizer' not in option_strings:
            group.add_argument('--use-precision-aware-optimizer', action='store_true', default=False,
                               help='Enable precision-aware optimizer state dtypes (exp_avg/exp_avg_sq).')
            group.add_argument('--main-grads-dtype', default='fp32', choices=sorted(_GRAD_TOKENS),
                               help='Dtype of main grads when enabling precision-aware optimizer.')
            group.add_argument('--main-params-dtype', default='fp32', choices=sorted(_MAIN_PARAM_TOKENS),
                               help='Dtype of master parameters when enabling precision-aware optimizer.')
            group.add_argument('--exp-avg-dtype', default='fp32', choices=sorted(_EXP_AVG_TOKENS),
                               help='Dtype of exp_avg when enabling precision-aware optimizer.')
            group.add_argument('--exp-avg-sq-dtype', default='fp32', choices=sorted(_EXP_AVG_SQ_TOKENS),
                               help='Dtype of exp_avg_sq when enabling precision-aware optimizer.')
        if '--quant-states' not in option_strings:
            group.add_argument('--quant-states', choices=_QUANT_STATE_CHOICES, default=None,
                               help='Select quantization format for optimizer states (default: disabled).')
        if '--quant-grads' not in option_strings:
            group.add_argument('--quant-grads', action='store_true',
                               help='Enable gradient quantization; dtype inferred from main_grads or defaults to fp16.')

    def validate_args(self, args):
        precision_enabled, _ = self._normalize_flags(args)
        if precision_enabled:
            self._validate_tokens(args)

    def register_patches(self, patch_manager, args):
        precision_enabled, quant_enabled = self._normalize_flags(args)

        if not (precision_enabled or quant_enabled):
            return

        patch_specs = []
        if quant_enabled:
            import mindspeed.core.optimizer.low_precision.quant_optimizer_hooks as optimizer_hooks
            import mindspeed.core.optimizer.low_precision.quant_distributed_hooks as distributed_hooks
            from mindspeed.core.optimizer.low_precision import quant_grad_clip as grad_clip
            from mindspeed.core.optimizer.low_precision import finalize_model_grads
            from mindspeed.core.optimizer.low_precision import distributed_data_parallel
            from mindspeed.core.optimizer.low_precision import param_and_grad_buffer
            patch_specs.extend(
                [
                    (
                        'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._collect_main_grad_data_for_unscaling',
                        distributed_hooks.collect_main_grad_data_for_unscaling_quant,
                        True,
                    ),
                    (
                        'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._copy_model_grads_to_main_grads',
                        distributed_hooks.copy_model_grads_to_main_grads_quant,
                        True,
                    ),
                ]
            )
        else:
            from mindspeed.core.optimizer.low_precision import optimizer_hooks, distributed_hooks
            from mindspeed.core.optimizer.low_precision import grad_clip

        patch_specs.extend(
            [
                (
                    'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.prepare_grads',
                    optimizer_hooks.prepare_grads_impl,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step_with_ready_grads',
                    optimizer_hooks.step_with_ready_grads_impl,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step',
                    optimizer_hooks.mixed_precision_optimizer_step_impl,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__',
                    optimizer_hooks.reuse_fp32_param_init_wrapper,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__',
                    optimizer_hooks.optimizer_config_init_wrapper,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer_config.OptimizerConfig.__post_init__',
                    optimizer_hooks.optimizer_config_post_init_wrapper,
                    True,
                ),
                (
                    'megatron.core.optimizer._get_megatron_optimizer_based_on_param_groups',
                    optimizer_hooks.get_optimizer_builder_wrapper,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params._collect_main_grad_data_for_unscaling',
                    optimizer_hooks.collect_main_grad_data_for_unscaling_wrapper,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params._copy_model_grads_to_main_grads',
                    optimizer_hooks.copy_model_grads_to_main_grads_wrapper,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan',
                    optimizer_hooks.unscale_main_grads_and_check_for_nan,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer.MegatronOptimizer.get_main_grads_for_grad_norm',
                    optimizer_hooks.get_main_grads_for_grad_norm,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer._zero_grad_group_helper',
                    optimizer_hooks.zero_grad_group_helper_wrapper,
                    True,
                ),
                (
                    'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._collect_main_grad_data_for_unscaling',
                    distributed_hooks.collect_main_grad_data_for_unscaling_wrapper,
                    True,
                ),
                (
                    'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._copy_model_grads_to_main_grads',
                    distributed_hooks.copy_model_grads_to_main_grads_wrapper,
                    True,
                ),
                (
                    'megatron.core.optimizer.clip_grads.get_grad_norm_fp32',
                    grad_clip.get_grad_norm_fp32,
                    True,
                ),
                (
                    'megatron.core.optimizer.clip_grads.clip_grad_by_total_norm_fp32',
                    grad_clip.clip_grad_by_total_norm_fp32_wrapper,
                    True,
                ),
            ]
        )

        if quant_enabled:
            from mindspeed.core.optimizer.low_precision import finalize_model_grads as quant_finalize
            patch_specs.extend(
                [
                    (
                        'megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_grad_sync',
                        param_and_grad_buffer.quant_grad_start_grad_sync_wrapper,
                        False,
                    ),
                    (
                        'megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.finish_grad_sync',
                        param_and_grad_buffer.quant_grad_finish_grad_sync_wrapper,
                        False,
                    ),
                    (
                        'megatron.core.distributed.param_and_grad_buffer._ParamAndGradBuffer.__init__',
                        param_and_grad_buffer.quant_grad_param_and_grad_buffer_init_wrapper,
                        False,
                    ),
                    (
                        'megatron.core.distributed.finalize_model_grads._allreduce_word_embedding_grads',
                        quant_finalize._allreduce_word_embedding_grads,
                        False,
                    ),
                    (
                        'megatron.core.distributed.finalize_model_grads._allreduce_position_embedding_grads',
                        quant_finalize._allreduce_position_embedding_grads,
                        False,
                    ),
                    (
                        'megatron.core.distributed.finalize_model_grads._allreduce_layernorm_grads',
                        quant_finalize._allreduce_layernorm_grads,
                        False,
                    ),
                    (
                        'megatron.core.distributed.finalize_model_grads._allreduce_conditional_embedding_grads',
                        quant_finalize._allreduce_conditional_embedding_grads,
                        False,
                    ),
                    (
                        'megatron.core.distributed.finalize_model_grads._update_router_expert_bias',
                        quant_finalize._update_router_expert_bias,
                        False,
                    ),
                    (
                        'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_backward_post_hook',
                        distributed_hooks.ddp_make_backward_post_hook_wrapper,
                        False,
                    ),
                    (
                        'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                        optimizer_hooks.distributed_optimizer_init_wrapper,
                        True,
                    ),
                ]
            )
            from mindspeed.core.models.gpt.gpt_model import gptmodel_init_wrapper
            from mindspeed.core.tensor_parallel.layers import copy_tensor_model_parallel_attributes_wrapper
            from mindspeed.core.optimizer.low_precision.language_model import transformer_language_model_init_wrapper
            patch_specs.extend(
                [
                    (
                        'megatron.core.models.gpt.gpt_model.GPTModel.__init__',
                        gptmodel_init_wrapper,
                        True,
                    ),
                    (
                        'megatron.core.tensor_parallel.layers.copy_tensor_model_parallel_attributes',
                        copy_tensor_model_parallel_attributes_wrapper,
                        True,
                    ),
                    (
                        'megatron.legacy.model.language_model.TransformerLanguageModel.__init__',
                        transformer_language_model_init_wrapper,
                        True,
                    ),
                ]
            )
        for target, func, force in patch_specs:
            patch_manager.register_patch(target, func, force_patch=force)
