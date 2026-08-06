# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MoEFixRouterFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('fix-router', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--fix-router", action='store_true', default=False, help='Use deterministic round-robin MoE expert routing.'
        )

    def validate_args(self, args):
        if args.fix_router and args.expert_model_parallel_size <= 1:
            raise AssertionError('when enabling fix-router, expert_model_parallel_size must be greater than 1')
        if args.fix_router and getattr(args, "moe_router_fusion", False):
            raise AssertionError("`--fix-router` does not support `--moe-router-fusion`")

    def register_patches(self, patch_manager, args):
        if args.fix_router:
            from mindspeed.core.transformer.moe.moe_utils import (
                compute_routing_scores_for_aux_loss_wrapper,
                topk_routing_with_score_function_wrapper,
            )

            patch_manager.register_patch(
                "megatron.core.transformer.moe.moe_utils.topk_routing_with_score_function",
                topk_routing_with_score_function_wrapper,
            )
            patch_manager.register_patch(
                "megatron.core.transformer.moe.moe_utils.compute_routing_scores_for_aux_loss",
                compute_routing_scores_for_aux_loss_wrapper,
            )
