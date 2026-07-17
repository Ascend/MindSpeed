from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MoEGmmFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('moe-grouped-gemm', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--gemm-gradient-accumulation-fusion", action='store_true', help="Use gradient-accumulation-fusion in gemm."
        )

    def validate_args(self, args):
        if getattr(args, 'moe_grouped_gemm', False):
            raise NotImplementedError(
                'MindSpeed --moe-grouped-gemm is not adapted to Megatron 0.18 in this scoped adaptation.'
            )
        if args.gemm_gradient_accumulation_fusion:
            if not args.moe_grouped_gemm:
                raise AssertionError('`--gemm-gradient-accumulation-fusion` only support with `--moe-grouped-gemm`.')

    def register_patches(self, patch_manager, args):
        # validate_args rejects this feature before patch registration. Keep this
        # method import-free so default Dense/basic-MoE startup never loads the
        # unadapted MindSpeed GMM or specialized MoE modules.
        return
