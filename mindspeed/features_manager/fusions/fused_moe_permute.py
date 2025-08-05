# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import warnings
from argparse import ArgumentParser, Namespace

import torch
import torch_npu

from mindspeed.features_manager.feature import MindSpeedFeature


class FusedMoEPermuteFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('moe-permute-fusion', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='fusion')

        group.add_argument("--use-fused-moe-token-permute-and-unpermute", action='store_true',
                           help="Use fused moe permute and unpermute.")

    def is_need_apply(self, args):
        return (self.optimization_level <= args.optimization_level and
                (getattr(args, self.feature_name, None)
                 or getattr(args, "use_fused_moe_token_permute_and_unpermute"), None))

    def validate_args(self, args: Namespace):
        warnings.warn(
            "Parameters --moe-permute-fusion and --use-fused-moe-token-permute-and-unpermute are equivalent. "
            "Use only one; prefer --moe-permute-fusion.")
        if args.use_fused_moe_token_permute_and_unpermute and not args.moe_permute_fusion:
            args.moe_permute_fusion = True
        if not args.use_fused_moe_token_permute_and_unpermute and args.moe_permute_fusion:
            args.use_fused_moe_token_permute_and_unpermute = True

    def pre_register_patches(self, pm, args):
        # The following patch is to pass the TransformerConfig.__post_init__ check
        pm.register_patch('transformer_engine.pytorch.permutation.moe_permute', torch.nn.Module, create_dummy=True)
        pm.register_patch('transformer_engine.pytorch.permutation.moe_permute_with_probs', torch.nn.Module,
                          create_dummy=True)
        pm.register_patch('transformer_engine.pytorch.permutation.moe_sort_chunks_by_index', torch.nn.Module,
                          create_dummy=True)
        pm.register_patch('transformer_engine.pytorch.permutation.moe_sort_chunks_by_index_with_probs',
                          torch.nn.Module,
                          create_dummy=True)
        pm.register_patch('transformer_engine.pytorch.permutation.moe_unpermute', torch.nn.Module,
                          create_dummy=True)

    def register_patches(self, patch_manager, args: Namespace):
        if getattr(args, self.feature_name, None) or getattr(args, "use_fused_moe_token_permute_and_unpermute", None):
            hasattr_npu_permute = hasattr(torch_npu, "npu_moe_token_permute_with_routing_map")
            hasattr_npu_unpermute = hasattr(torch_npu, "npu_moe_token_unpermute_with_routing_map")
            if not hasattr_npu_permute:
                raise AttributeError(
                    "torch_npu should have attribute npu_moe_token_permute_with_routing_map, but "
                    "does not have it. Please upgrade CANN to 8.3.RC1 and later, and PTA to 7.2.RC1 and later")
            if not hasattr_npu_unpermute:
                raise AttributeError(
                    "torch_npu should have attribute npu_moe_token_unpermute_with_routing_map, but "
                    "does not have it. Please upgrade CANN to 8.3.RC1 and later, and PTA to 7.2.RC1 and later")

            from mindspeed.core.fusions.fused_moe_permute import permute, unpermute, sort_chunks_by_idxs_wrapper, \
                moe_alltoall_token_dispatcher_init_wrapper, maybe_dtoh_and_synchronize
            # Since te permute interface lacks the input parameter drop_and_pad required by
            # npu_moe_token_permute_with_routing_map, the te interface cannot be directly replaced. Instead, the
            # megatron permute interface must be replaced.
            patch_manager.register_patch('megatron.core.transformer.moe.moe_utils.permute', permute)

            # Since te unpermute interface lacks the input parameter routing_map required by
            # npu_moe_token_unpermute_with_routing_map, the te interface cannot be directly replaced. Instead, the
            # megatron unpermute interface must be replaced.
            patch_manager.register_patch('megatron.core.transformer.moe.moe_utils.unpermute', unpermute)

            warnings.warn("Currently, fused_sort_chunks_by_index is not supported")
            patch_manager.register_patch('megatron.core.transformer.moe.moe_utils.sort_chunks_by_idxs',
                                         sort_chunks_by_idxs_wrapper)

            # Since fused_sort_chunks_by_index is not currently supported, when self.num_local_experts > 1,
            # move self.num_global_tokens_per_local_expert to cpu
            patch_manager.register_patch(
                'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize',
                maybe_dtoh_and_synchronize)

            # Since fused_sort_chunks_by_index is not currently supported, set self.permute_idx_device to None
            patch_manager.register_patch(
                'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.__init__',
                moe_alltoall_token_dispatcher_init_wrapper)
