from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MoEGmmFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('moe-grouped-gemm', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--gemm-gradient-accumulation-fusion", action='store_true',
                           help="Use gradient-accumulation-fusion in gemm.")

    def validate_args(self, args):
        if args.gemm_gradient_accumulation_fusion:
            if not args.moe_grouped_gemm:
                raise AssertionError('`--gemm-gradient-accumulation-fusion` only support with `--moe-grouped-gemm`.')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedGmmExperts
        if args.moe_grouped_gemm:
            patch_manager.register_patch(
                'megatron.core.transformer.moe.experts.GroupedMLP',
                MindSpeedGmmExperts)
        if args.use_ascend_mc2 and not hasattr(args, 'moe_grouped_gemm'):
            # MoE MLP not use mc2 linear
            from mindspeed.core.models.gpt.gpt_layer_specs import build_layers_wrapper
            from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
            from megatron.core.transformer.transformer_block import TransformerBlock
            TransformerBlock._build_layers = build_layers_wrapper(TransformerBlock._build_layers,
                                                                  ColumnParallelLinear.forward,
                                                                  RowParallelLinear.forward)

        # TEGroupedMLP performance.
        from mindspeed.te.pytorch.module.grouped_linear import mindspeed_groupedmlp_weighted_bias_swiglu_impl
        patch_manager.register_patch(
            'megatron.core.fusions.fused_bias_swiglu.weighted_bias_swiglu_impl',
            mindspeed_groupedmlp_weighted_bias_swiglu_impl
        )
