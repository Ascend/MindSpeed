# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from argparse import ArgumentParser
import torch
from mindspeed.features_manager.feature import MindSpeedFeature


class MoEAlltoAllSeqOverLapFeature(MindSpeedFeature):
    '''
    MoE Layer AllToAllSeq OverLap spec.
    This spec supports "alltoall_seq" dispatcher.
    '''
    def __init__(self):
        super().__init__('moe-alltoall-overlap-comm', 2)
 
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--moe-alltoall-overlap-comm', action='store_true', default=False,
                        help='Use async communication&swap to overlap compute in alltoall_seq.')
        group.add_argument("--moe-zero-memory", type=str, default='disable',
                        choices=['disable', 'level0', 'level1'],
                        help="Set level for saving activation memory in moe layer.")
        group.add_argument('--moe-zero-memory-num-layers', type=int, default=None,
                        help='the number of layers using moe-zero-memory level1'
                                'in each pp stage.')

    def validate_args(self, args):
        self.incompatible_check(args, 'use_ascend_mc2')
        if args.moe_alltoall_overlap_comm and not args.moe_token_dispatcher_type == 'alltoall_seq':
            raise AssertionError('`--moe-alltoall-overlap-comm` only support with `--moe-token-dispatcher-type alltoall_seq`.')
        if args.moe_alltoall_overlap_comm:
            if not args.moe_permutation_async_comm:
                raise AssertionError('`--moe-alltoall-overlap-comm` and `--moe-allgather-overlap-comm` only support with `--moe-permutation-async-comm`.')
            if not args.moe_grouped_gemm:
                raise AssertionError('`--moe-alltoall-overlap-comm` and `--moe-allgather-overlap-comm` only support with `--moe-grouped-gemm`.')
        if not args.moe_tp_extend_ep and args.moe_alltoall_overlap_comm:
            raise AssertionError('`--moe-alltoall-overlap-comm` needs `moe_tp_extend_ep`.')
        #Share Experts convert & check.
        if args.n_shared_experts is not None and args.moe_shared_expert_intermediate_size is None:
            args.moe_shared_expert_intermediate_size = args.n_shared_experts * args.ffn_hidden_size
            print(f'Using shared experts. Convert n_shared_experts to moe_shared_expert_intermediate_size, the moe_shared_expert_intermediate_size is {args.moe_shared_expert_intermediate_size}.')
        elif args.n_shared_experts is None and args.moe_shared_expert_intermediate_size is not None:
            args.n_shared_experts = args.moe_shared_expert_intermediate_size // args.ffn_hidden_size
            print(f'Using shared experts. Convert moe_shared_expert_intermediate_size to n_shared_experts, the n_shared_experts is {args.n_shared_experts}.')
        #Zero Memory check.
        if args.moe_zero_memory_num_layers is not None:
            num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
            if args.moe_zero_memory_num_layers < 0 or args.moe_zero_memory_num_layers > num_layers_per_pipeline_stage:
                raise AssertionError('`--moe-zero-memory-num-layers` must be between 0 and num layers per pipeline stage')
            if args.moe_zero_memory == "disable":
                raise AssertionError('`--moe-zero-memory` must be enabled when using `--moe-zero-memory-num-layers`')
        if args.moe_zero_memory != "disable" and args.moe_allgather_overlap_comm:
            raise AssertionError('`--moe-zero-memory` do not support `--moe-allgather-overlap-comm` for now.')
        
    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedAlltoAllSeqOverlapMoeLayerAdaptor
        from mindspeed.core.transformer.moe.moe_feature.overlap.moe_common import mlp_init, parallel_transformer_layer_init_wrapper, core_mlp_forward_wrapper
        from mindspeed.core.transformer.moe.moe_feature.overlap.experts import Zero_Memory_SharedExpertMlp_forward
        patch_manager.register_patch('megatron.core.transformer.mlp.MLP.forward',
                    core_mlp_forward_wrapper)
        if getattr(args, 'moe_token_dispatcher_type', None) == "alltoall_seq":
            if args.moe_alltoall_overlap_comm:
                patch_manager.register_patch(
                    'megatron.core.transformer.mlp.MLP.__init__',
                    mlp_init)
                patch_manager.register_patch(
                    'megatron.core.transformer.transformer_layer.TransformerLayer.__init__',
                    parallel_transformer_layer_init_wrapper)
                patch_manager.register_patch(
                    'megatron.core.transformer.moe.moe_layer.MoELayer', 
                    MindSpeedAlltoAllSeqOverlapMoeLayerAdaptor)
                if args.moe_zero_memory != 'disable':
                    patch_manager.register_patch(
                        'megatron.core.transformer.moe.shared_experts.SharedExpertMLP.forward',
                        Zero_Memory_SharedExpertMlp_forward)
