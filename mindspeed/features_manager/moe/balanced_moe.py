# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class BalancedMoEFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('balanced-moe-experts')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--balanced-moe-experts", action='store_true', default=False,
                           help='Enable balanced MoE Experts Balance workload across EPs by duplicating experts.')
        group.add_argument('--balanced-moe-hot-expert-num', type=int, default=3,
                           help='The number of duplicated hot experts to balance MoE workloads.')
        group.add_argument('--trans-hot-expert-group-num', type=int, default=4,
                           help='trans hot expert group num')

    def validate_args(self, args):
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
