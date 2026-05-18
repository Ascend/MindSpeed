# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class GDNFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('gated-delta-net')
        
    def is_need_apply(self, args):
        """Check the feature is need to apply."""
        return (self.optimization_level <= args.optimization_level and getattr(args, self.feature_name, 1)) \
            or self.default_patches

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--use-naive-l2norm", action='store_true',
                           help="")

    def register_patches(self, patch_manager, args):
        _use_cp = int(getattr(args, 'context_parallel_size', 1)) > 1
        _cp_algo = getattr(args, 'context_parallel_algo', 'megatron_cp_algo')
        _cp_expanded_by_2d_tp = getattr(args, 'tp_2d', False) and getattr(args, 'tp_y', 1) > 1
        _use_nave_l2norm = getattr(args, 'use_naive_l2norm', False)

        use_core_gdn = _use_cp or (_cp_expanded_by_2d_tp and _cp_algo == 'megatron_cp_algo') or _use_nave_l2norm
        if use_core_gdn:
            # gdn feature
            from mindspeed.core.ssm.gated_delta_net import GatedDeltaNet
            patch_manager.register_patch('megatron.core.ssm.gated_delta_net.GatedDeltaNet', GatedDeltaNet)