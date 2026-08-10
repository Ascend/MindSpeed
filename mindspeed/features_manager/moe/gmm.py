from argparse import ArgumentParser
from copy import copy
from dataclasses import replace
from functools import partial, wraps
from inspect import signature

from mindspeed.features_manager.feature import MindSpeedFeature


def _uses_grouped_gemm(fn, args, kwargs):
    """Return whether this individual Megatron 0.18 spec request enables grouped GEMM."""
    bound_args = signature(fn).bind_partial(*args, **kwargs)
    return bool(bound_args.arguments.get('moe_grouped_gemm', False))


def _replace_moe_experts_builder(spec, experts_builder):
    """Replace routed experts in a Megatron 0.18 functools.partial MoE builder."""
    if not isinstance(spec, partial):
        raise TypeError(f'Megatron v0.18 MoE spec must be a functools.partial MlpBuilder, but got {type(spec)!r}.')
    submodules = (spec.keywords or {}).get('submodules')
    if submodules is None:
        raise TypeError("Megatron v0.18 MoE spec must provide partial.keywords['submodules'].")

    new_keywords = dict(spec.keywords or {})
    new_keywords['submodules'] = replace(submodules, experts=experts_builder)
    new_spec = partial(spec.func, *spec.args, **new_keywords)
    new_spec.__dict__.update(
        {key: value for key, value in spec.__dict__.items() if key not in {'module', 'submodules'}}
    )
    return new_spec


def _build_te_grouped_mlp(experts_builder, num_local_experts, config, *args, **kwargs):
    """Build Megatron TEGroupedMLP with expert-only gradient fusion settings."""
    expert_config = copy(config)
    expert_config.gradient_accumulation_fusion = bool(getattr(config, 'gemm_gradient_accumulation_fusion', False))
    return experts_builder(num_local_experts, expert_config, *args, **kwargs)


def get_moe_module_spec_gmm_wrapper(fn):
    """Keep Megatron TEGroupedMLP and adapt its expert-only configuration."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        spec = fn(*args, **kwargs)
        if not _uses_grouped_gemm(fn, args, kwargs):
            return spec

        from megatron.core.transformer.moe.experts import TEGroupedMLP

        submodules = (spec.keywords or {}).get('submodules')
        if submodules is None:
            raise TypeError("Megatron v0.18 MoE spec must provide partial.keywords['submodules'].")
        experts_builder = submodules.experts
        experts_impl = experts_builder.func if isinstance(experts_builder, partial) else experts_builder
        if experts_impl is not TEGroupedMLP:
            raise TypeError(
                'MindSpeed --moe-grouped-gemm requires the Transformer Engine '
                f'TEGroupedMLP builder, but got {experts_impl!r}.'
            )

        return _replace_moe_experts_builder(
            spec,
            partial(_build_te_grouped_mlp, experts_builder),
        )

    return wrapper


def te_grouped_linear_init_wrapper(fn):
    """Adapt Megatron's TEGroupedLinear construction for TE-NPU grouped weights."""
    fn_signature = signature(fn)

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        bound_args = fn_signature.bind(self, *args, **kwargs)
        parallel_mode = bound_args.arguments.get('parallel_mode')
        config = bound_args.arguments.get('config')

        if getattr(config, 'gemm_gradient_accumulation_fusion', False):
            config.moe_single_grouped_weight = True

        fn(self, *args, **kwargs)

        grouped_weight = getattr(self, 'weight', None)
        if grouped_weight is not None and parallel_mode in ('column', 'row'):
            # Packed TE weights use [expert, output, input].
            grouped_weight.partition_dim = 1 if parallel_mode == 'column' else 2
            grouped_weight.partition_stride = 1

    return wrapper


def tenpu_grouped_linear_init_wrapper(fn):
    """Use TE-NPU's packed-weight backend for fused grouped wgrad accumulation."""
    fn_signature = signature(fn)

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        bound_args = fn_signature.bind(self, *args, **kwargs)
        if bound_args.arguments.get('fuse_wgrad_accumulation', False):
            bound_args.arguments['single_grouped_weight'] = True
        return fn(*bound_args.args, **bound_args.kwargs)

    return wrapper


class MoEGmmFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('moe-grouped-gemm', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--gmm-gradient-accumulation-fusion",
            "--gemm-gradient-accumulation-fusion",
            dest="gemm_gradient_accumulation_fusion",
            action='store_true',
            help="Use gradient-accumulation-fusion in GMM.",
        )

    def validate_args(self, args):
        if args.gemm_gradient_accumulation_fusion:
            if not args.moe_grouped_gemm:
                raise AssertionError('`--gmm-gradient-accumulation-fusion` only support with `--moe-grouped-gemm`.')
            if getattr(args, 'fp8', None) or getattr(args, 'fp4', None):
                raise NotImplementedError(
                    'MindSpeed grouped GEMM gradient accumulation fusion currently '
                    'supports only non-FP8/non-FP4 training.'
                )
            if not (getattr(args, 'fp16', False) or getattr(args, 'bf16', False)):
                raise NotImplementedError(
                    'MindSpeed grouped GEMM gradient accumulation fusion currently requires FP16 or BF16 activations.'
                )
        if args.moe_grouped_gemm and args.transformer_impl != 'transformer_engine':
            raise AssertionError('MindSpeed --moe-grouped-gemm requires --transformer-impl transformer_engine.')

    def register_patches(self, patch_manager, args):
        if args.moe_grouped_gemm:
            patch_manager.register_patch(
                'megatron.core.models.gpt.moe_module_specs.get_moe_module_spec_for_backend',
                get_moe_module_spec_gmm_wrapper,
            )
            patch_manager.register_patch(
                'megatron.core.extensions.transformer_engine.TEGroupedLinear.__init__',
                te_grouped_linear_init_wrapper,
            )
            patch_manager.register_patch(
                'transformer_engine.pytorch.module.grouped_linear.GroupedLinear.__init__',
                tenpu_grouped_linear_init_wrapper,
            )

        if args.use_ascend_mc2 and not hasattr(args, 'moe_grouped_gemm'):
            # MoE MLP not use mc2 linear
            from mindspeed.core.models.gpt.gpt_layer_specs import build_layers_wrapper
            from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
            from megatron.core.transformer.transformer_block import TransformerBlock

            TransformerBlock._build_layers = build_layers_wrapper(
                TransformerBlock._build_layers, ColumnParallelLinear.forward, RowParallelLinear.forward
            )

        # TEGroupedMLP performance.
        from mindspeed.core.transformer.moe.grouped_matmul_util import mindspeed_groupedmlp_weighted_bias_swiglu_impl

        patch_manager.register_patch(
            'megatron.core.fusions.fused_bias_swiglu.weighted_bias_swiglu_impl',
            mindspeed_groupedmlp_weighted_bias_swiglu_impl,
        )
