# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature
MAX_GROUP_NUM = 8


class BalancedMoEFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('balanced-moe-experts')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--balanced-moe-experts", action='store_true', default=False,
                           help='Enable balanced MoE Experts Balance workload across EPs by duplicating experts.')
        group.add_argument('--balanced-moe-hot-expert-num', type=int, default=3,
                           help='The number of duplicated hot experts to balance MoE workloads.')
        group.add_argument('--trans-hot-expert-group-num', type=int, default=3,
                           help='trans hot expert group num')

    def validate_args(self, args):
        # 1. Check if balanced_moe_experts is enabled
        if not getattr(args, 'balanced_moe_experts', False):
            return

        # 2. Validate hot expert count N
        if args.balanced_moe_hot_expert_num <= 0:
            raise ValueError(
                f"--balanced-moe-hot-expert-num must be positive, got {args.balanced_moe_hot_expert_num}"
            )

        # 3. Validate hot expert count does not exceed local experts
        num_local_experts = args.num_experts // args.expert_model_parallel_size
        if args.balanced_moe_hot_expert_num > num_local_experts:
            raise ValueError(
                f"--balanced-moe-hot-expert-num ({args.balanced_moe_hot_expert_num}) "
                f"must be <= num_local_experts ({num_local_experts}) "
                f"(where num_local_experts = num_experts / expert_model_parallel_size = "
                f"{args.num_experts} / {args.expert_model_parallel_size})"
            )

        # 4. Validate transmission group count M
        if args.trans_hot_expert_group_num <= 0:
            raise ValueError(
                f"--trans-hot-expert-group-num must be positive, got {args.trans_hot_expert_group_num}"
            )

        # 5. If transmission group count exceeds hot expert count, warn and auto-adjust
        if args.trans_hot_expert_group_num > args.balanced_moe_hot_expert_num:
            print(f"⚠️ Warning: --trans-hot-expert-group-num ({args.trans_hot_expert_group_num}) "
                  f"is greater than --balanced-moe-hot-expert-num ({args.balanced_moe_hot_expert_num}). "
                  f"Automatically adjusting to {args.balanced_moe_hot_expert_num}.")
            args.trans_hot_expert_group_num = args.balanced_moe_hot_expert_num

        # 6. Check if group count exceeds maximum limit (if needed)
        if args.trans_hot_expert_group_num > MAX_GROUP_NUM:
            print(f"⚠️ Warning: --trans-hot-expert-group-num ({args.trans_hot_expert_group_num}) "
                  f"is greater than default MAX_GROUP_NUM  ({MAX_GROUP_NUM}). "
                  f"Automatically adjusting to {MAX_GROUP_NUM}.")
            args.trans_hot_expert_group_num = MAX_GROUP_NUM

        # 7. Provide recommendations based on EP size
        ep_size = args.expert_model_parallel_size
        if ep_size >= 32:
            print(f"  - ✓ Good: EP size ({ep_size}) is large enough for optimal performance benefits.")
        elif ep_size >= 16:
            print(f"  - ⚠️ Moderate: EP size ({ep_size}) is moderate. Benefits may be limited.")
        else:
            print(
                f"  - ⚠️ Caution: EP size ({ep_size}) is small. Load balancing benefits may not justify communication overhead.")

        self.dependency_check(args, 'moe_fb_overlap')
        self.dependency_check(args, 'moe_grouped_gemm')
        if getattr(args, 'balanced_moe_experts', False) and getattr(args, 'moe_token_dispatcher_type', None) != "alltoall":
            raise AssertionError('Currently, --balanced-moe-experts only support alltoall token dispatcher')
        self.incompatible_check(args, 'moe_expert_capacity_factor')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_feature.balanced_moe.modules.moe_layer import BalancedMoELayer
        from mindspeed.core.transformer.moe.moe_feature.balanced_moe.adaptor import get_moe_module_spec_wrapper, \
            mindspeed_initialize_model_parallel_wrapper
        patch_manager.register_patch('megatron.core.models.gpt.moe_module_specs.get_moe_module_spec',
                                     get_moe_module_spec_wrapper)
        patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                                     mindspeed_initialize_model_parallel_wrapper)
