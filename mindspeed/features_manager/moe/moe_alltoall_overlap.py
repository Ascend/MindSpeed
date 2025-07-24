# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser
import torch
from mindspeed.features_manager.feature import MindSpeedFeature


class MoEAlltoAllOverLapFeature(MindSpeedFeature):
    '''
    MoE Layer AllToAll or alltoall_seq OverLap spec.
    This spec supports "alltoall" and "alltoall_seq" dispatcher.
    '''
    def __init__(self):
        super().__init__('moe-alltoall-overlap-comm', 2)
 
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--moe-alltoall-overlap-comm', action='store_true', default=False,
                        help='Use async communication&swap to overlap compute in alltoall or alltoall_seq. In alltoall dispatcher, \
                        if with share_expert, will open `--moe-shared-expert-overlap` automatically.')

    def validate_args(self, args):
        self.incompatible_check(args, 'use_ascend_mc2')
        if args.moe_alltoall_overlap_comm and args.moe_token_dispatcher_type not in ('alltoall', 'alltoall_seq'):
            raise AssertionError('`--moe-alltoall-overlap-comm` only support with `--moe-token-dispatcher-type alltoall` or `--moe-token-dispatcher-type alltoall_seq`.')
        if args.moe_alltoall_overlap_comm:
            if args.moe_token_dispatcher_type == 'alltoall':
                if not args.moe_grouped_gemm:
                    raise AssertionError('`--moe-alltoall-overlap-comm` and `--moe-allgather-overlap-comm` only support with `--moe-grouped-gemm`.')
                if args.moe_tp_extend_ep:
                    raise AssertionError('`alltoall` not support `--moe-tp-extend-ep` for now. With`--moe-tp-extend-ep`, the dispatcher should be `alltoall_seq`.')
                if (args.n_shared_experts or args.moe_shared_expert_intermediate_size) and not args.moe_shared_expert_overlap:
                    args.moe_shared_expert_overlap = True
                    print('Warning: with `alltoall` dispatcher and share_expert, open `--moe-shared-expert-overlap`.')

            elif args.moe_token_dispatcher_type == 'alltoall_seq':
                if not args.moe_permutation_async_comm:
                    raise AssertionError('`--moe-alltoall-overlap-comm` with `alltoall_seq` dispatcher needs `--moe-permutation-async-comm`.')
                if not args.moe_grouped_gemm:
                    raise AssertionError('`--moe-alltoall-overlap-comm` with `alltoall_seq` dispatcher needs `--moe-grouped-gemm`.')
                if not args.moe_tp_extend_ep and args.moe_alltoall_seq_overlap_comm:
                    raise AssertionError('`--moe-alltoall-overlap-comm` with `alltoall_seq` needs `moe_tp_extend_ep`.')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedAlltoAllOverlapMoeLayerAdaptor, MindSpeedAlltoAllSeqOverlapMoeLayerAdaptor
        from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import mlp_init, parallel_transformer_layer_init_wrapper, core_mlp_forward_wrapper
        patch_manager.register_patch('megatron.core.transformer.mlp.MLP.forward',
                    core_mlp_forward_wrapper)
        if hasattr(args, 'moe_token_dispatcher_type') and args.moe_alltoall_overlap_comm:
            patch_manager.register_patch(
                'megatron.core.transformer.mlp.MLP.__init__',
                mlp_init)
            patch_manager.register_patch(
                'megatron.core.transformer.transformer_layer.TransformerLayer.__init__',
                parallel_transformer_layer_init_wrapper)
            if args.moe_token_dispatcher_type == 'alltoall':
                patch_manager.register_patch(
                    'megatron.core.transformer.moe.moe_layer.MoELayer', 
                    MindSpeedAlltoAllOverlapMoeLayerAdaptor)
            elif args.moe_token_dispatcher_type == 'alltoall_seq':
                patch_manager.register_patch(
                    'megatron.core.transformer.moe.moe_layer.MoELayer', 
                    MindSpeedAlltoAllSeqOverlapMoeLayerAdaptor)