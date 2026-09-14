# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MoENpuFastOpsFeature(MindSpeedFeature):
    """Default-on NPU fast-path op selection for MoE permute / unpermute / routing.

    Restores the mcore 0.12 op selection (``masked_select`` permute, ``scatter``
    routing map, ``scatter_add_`` unpermute) to avoid the AICPU ``aclnnSort`` /
    ``aclnnIndexPutImpl`` kernels that Megatron 0.15+/0.17+ selects on NPU. The
    routing patch stays active with ``--moe-permute-fusion`` (whose fused NPU
    operators replace permute/unpermute themselves), and the whole feature can be
    disabled with ``--no-moe-npu-fast-ops``.
    """

    def __init__(self):
        super().__init__('moe-npu-fast-ops', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            '--no-moe-npu-fast-ops',
            action='store_false',
            dest='moe_npu_fast_ops',
            default=True,
            help='Disable the NPU fast-path MoE ops (masked_select permute, scatter-based routing '
            'map, scatter_add_ unpermute) and fall back to the Megatron op selection.',
        )

    def is_need_apply(self, args):
        return getattr(args, self.feature_name, True)

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_npu_fast_ops import (
            permute_npu,
            topk_routing_with_score_function_scatter_wrapper,
            unpermute_npu,
        )

        patch_manager.register_patch(
            'megatron.core.transformer.moe.moe_utils.topk_routing_with_score_function',
            topk_routing_with_score_function_scatter_wrapper,
        )

        if getattr(args, 'moe_permute_fusion', False) or getattr(
            args, 'use_fused_moe_token_permute_and_unpermute', False
        ):
            # FusedMoEPermuteFeature replaces permute/unpermute with fused NPU
            # operators, so its patches own those two targets.
            return

        patch_manager.register_patch('megatron.core.transformer.moe.moe_utils.permute', permute_npu)
        patch_manager.register_patch('megatron.core.transformer.moe.moe_utils.unpermute', unpermute_npu)
