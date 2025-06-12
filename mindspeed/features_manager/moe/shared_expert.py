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
        # Use megatron shared_experts replace.
        # Share Experts convert & check.
        if args.n_shared_experts and args.moe_shared_expert_intermediate_size:
            raise AssertionError('`n_shared_experts` cannot be used with `moe_shared_expert_intermediate_size` together. Please use one of them.')
        if args.n_shared_experts and args.moe_shared_expert_intermediate_size is None:
            args.moe_shared_expert_intermediate_size = args.n_shared_experts * args.ffn_hidden_size
            print(f'Using shared experts. Convert n_shared_experts to moe_shared_expert_intermediate_size, the moe_shared_expert_intermediate_size is {args.moe_shared_expert_intermediate_size}.')
        elif args.n_shared_experts is None and args.moe_shared_expert_intermediate_size:
            args.n_shared_experts = args.moe_shared_expert_intermediate_size // args.ffn_hidden_size
            print(f'Using shared experts. Convert moe_shared_expert_intermediate_size to n_shared_experts, the n_shared_experts is {args.n_shared_experts}.')
