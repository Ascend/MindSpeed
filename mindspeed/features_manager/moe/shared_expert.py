from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MoESharedExpertsFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('n-shared-experts', 0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--n-shared-experts', type=int, default=None,
                           help='DEPRECATED. use --moe-shared-expert-intermediate-size replace')

    def pre_validate_args(self, args):
        # use megatron shared_experts replace
        if args.n_shared_experts and args.moe_shared_expert_intermediate_size is None:
            args.moe_shared_expert_intermediate_size = args.n_shared_experts * args.moe_ffn_hidden_size