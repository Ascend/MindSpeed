# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MoEZeroMemoryFeature(MindSpeedFeature):
    '''
    Zero-Memory Settings spec.
    This spec supports "alltoall" and "alltoallseq" dispatcher.
    '''
    def __init__(self):
        super().__init__('moe-zero-memory', 2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--moe-zero-memory", type=str, default='disable',
                        choices=['disable', 'level0', 'level1'],
                        help="Set level for saving activation memory in moe layer.")
        group.add_argument('--moe-zero-memory-num-layers', type=int, default=None,
                        help='the number of layers using moe-zero-memory level1'
                                'in each pp stage.')

    def pre_validate_args(self, args):
        #Zero Memory check.
        if args.moe_zero_memory_num_layers is not None:
            num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
            if args.moe_zero_memory_num_layers < 0 or args.moe_zero_memory_num_layers > num_layers_per_pipeline_stage:
                raise AssertionError('`--moe-zero-memory-num-layers` must be between 0 and num layers per pipeline stage')
            if args.moe_zero_memory == "disable":
                raise AssertionError('`--moe-zero-memory` must be enabled when using `--moe-zero-memory-num-layers`')
        if args.moe_zero_memory != "disable" and not (args.moe_alltoall_overlap_comm or args.moe_fb_overlap):
            raise AssertionError('`--moe-zero-memory` only support `--moe-alltoall-overlap-comm` or `--moe-fb-overlap` for now.')

    def register_patches(self, patch_manager, args):
        if args.moe_zero_memory != 'disable' and args.moe_alltoall_overlap_comm:
            from mindspeed.core.transformer.moe.moe_feature.overlap.experts import zero_memory_shared_expert_mlp_forward
            patch_manager.register_patch(
                'megatron.core.transformer.moe.shared_experts.SharedExpertMLP.forward',
                zero_memory_shared_expert_mlp_forward)
