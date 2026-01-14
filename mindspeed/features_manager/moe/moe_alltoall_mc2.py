# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class MoEAlltoAllMC2Feature(MindSpeedFeature):
    '''
    MoE Layer AllToAll or alltoall_seq MC2 spec.
    This spec supports "alltoall" and "alltoall_seq" dispatcher.
    '''
    def __init__(self):
        super().__init__('moe-alltoall-mc2', 2)
 
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--moe-alltoall-mc2', action='store_true', default=False,
                        help='[expert] Use MC2 fused kernal for moe layer when dispatcher is alltoall_seq and alltoall. \
                        if with share_expert, will open `--moe-shared-expert-overlap` automatically for now.')

    def validate_args(self, args):
        self.incompatible_check(args, 'use_ascend_mc2')
        self.incompatible_check(args, 'moe_alltoall_overlap_comm')
        self.incompatible_check(args, 'moe_allgather_overlap_comm')
        self.incompatible_check(args, 'moe_fb_overlap')
        self.incompatible_check(args, 'moe_tp_extend_ep')

        if args.moe_alltoall_mc2 and args.moe_token_dispatcher_type not in ('alltoall_seq', 'alltoall'):
            raise AssertionError('`--moe-alltoall-mc2` only support with `--moe-token-dispatcher-type alltoall` or `--moe-token-dispatcher-type alltoall_seq`.')

        if args.moe_alltoall_mc2:

            if args.moe_token_dispatcher_type == 'alltoall_seq':
                if args.tensor_model_parallel_size != 1:
                    raise AssertionError('--moe-alltoall-mc2` with `alltoall_seq` needs TP=1.')
            elif args.moe_token_dispatcher_type == 'alltoall': 
                if args.expert_tensor_parallel_size != 1:
                    raise AssertionError('--moe-alltoall-mc2` with `alltoall_seq` needs ETP=1.')
                if args.moe_expert_capacity_factor is not None:
                    raise AssertionError('--moe-alltoall-mc2` only support dropless for now.')
            else:
                raise AssertionError('dispatcher type error! Please check!!!')

        if args.moe_alltoall_mc2 and args.moe_shared_expert_intermediate_size is not None:
            Warning('`--moe-alltoall-mc2` with `--moe-shared-expert-intermediate-size` will use `--moe-shared-expert-overlap` for now.')
            if args.moe_token_dispatcher_type == 'alltoall':
                # In alltoall_seq dispatcher, set moe_shared_expert_overlap in MoeLayer to bypass megatron's check.
                args.moe_shared_expert_overlap = True

        # Convert Megatron Shared_experts to MindSpeed version. This convert operation only for some judge.
        if args.n_shared_experts is None and args.moe_shared_expert_intermediate_size is not None:
            args.n_shared_experts = args.moe_shared_expert_intermediate_size // (
                args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size)

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedAlltoAllMC2MoeLayerAdaptor
        if hasattr(args, 'moe_token_dispatcher_type') and getattr(args, "moe_alltoall_mc2", False):

            patch_manager.register_patch(
                'megatron.core.transformer.moe.moe_layer.MoELayer', 
                MindSpeedAlltoAllMC2MoeLayerAdaptor)