#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MoeZeRCFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('moe-zerc')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--moe-zerc', action='store_true',
                           help='moe zero redundancy communication')

    def validate_args(self, args):
        if args.moe_zerc:
            tp_extended_ep_size = args.expert_model_parallel_size
            if args.moe_tp_extend_ep:
                tp_extended_ep_size *= args.tensor_model_parallel_size
            if tp_extended_ep_size == 1:
                raise AssertionError(
                    "moe_zerc is only supported with tp_extended_ep_size > 1")                
            num_local_experts = args.num_experts // tp_extended_ep_size
            if num_local_experts == 1:
                raise AssertionError(
                    "moe_zerc is only supported with num_local_experts > 1")
            self.incompatible_check(args, 'moe_pad_expert_input_to_capacity.')
            if args.schedules_method != "dualpipev":
                raise AssertionError(
                    "moe_zerc currently requires dualpipev.")
            self.incompatible_check(args, 'moe_unperm2_mem_optim_swap')

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_zerc.token_dispatcher import zerc_alltoall_token_perm1, \
            zerc_alltoall_token_perm2, zerc_alltoall_token_unperm1, zerc_alltoall_token_unperm2
        from mindspeed.core.transformer.moe.moe_zerc.fwdbwd import (
            transformer_layer_forward_moe_backward_moe_overlaping_zerc, 
            transformer_layer_forward_moe_backward_dense_overlaping_zerc
        )
        if args.moe_zerc:
            patch_manager.register_patch(
                'mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.alltoall_token_perm1',
                zerc_alltoall_token_perm1)
            patch_manager.register_patch(
                'mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.alltoall_token_perm2',
                zerc_alltoall_token_perm2)
            patch_manager.register_patch(
                'mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.alltoall_token_unperm1',
                zerc_alltoall_token_unperm1)
            patch_manager.register_patch(
                'mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.alltoall_token_unperm2',
                zerc_alltoall_token_unperm2)
            patch_manager.register_patch(
                'mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_moe_backward_moe_overlaping',
                transformer_layer_forward_moe_backward_moe_overlaping_zerc)
            patch_manager.register_patch(
                'mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_moe_backward_dense_overlaping',
                transformer_layer_forward_moe_backward_dense_overlaping_zerc)
