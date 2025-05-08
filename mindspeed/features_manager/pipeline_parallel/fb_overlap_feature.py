#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class FwdBwdOverlapFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('moe-fb-overlap')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--moe-fb-overlap', action='store_true')
        group.add_argument('--moe-unperm2-mem-optim', action='store_true')
        group.add_argument('--moe-unperm2-mem-optim-swap', action='store_true')



    def validate_args(self, args):
        self.incompatible_check(args, 'moe_alltoall_overlap_comm')
        self.incompatible_check(args, 'overlap_grad_reduce')
        self.incompatible_check(args, 'moe_hierarchical_alltoallv')
        self.incompatible_check(args, 'moe_zero_memory_num_layers')
        self.incompatible_check(args, 'moe_expert_capacity_factor')
        self.incompatible_check(args, 'use_nanopipe')
        self.incompatible_check(args, 'automated_pipeline')
        self.incompatible_check(args, 'recompute_in_bubble')
        self.incompatible_check(args, 'recompute_in_advance')
        self.incompatible_check(args, 'use_legacy_models')
        if args.moe_fb_overlap and args.moe_token_dispatcher_type == 'allgather':
            raise AssertionError('The fb overlap feature do not support allgather dispatcher.')
        if args.moe_fb_overlap and args.moe_zero_memory == 'level1':
            raise AssertionError('fb overlap only support moe zero memory level 0.')

        self.dependency_check(args, 'n_shared_experts')
        self.dependency_check(args, 'moe_permutation_async_comm')
        if args.tensor_model_parallel_size > 1:
            self.dependency_check(args, 'moe_tp_extend_ep')
        if args.moe_unperm2_mem_optim and not args.moe_fb_overlap:
            raise AssertionError('--moe-unperm2-mem-optim currently only can be used with --moe-fb-overlap')
        if args.moe_unperm2_mem_optim_swap and not args.moe_fb_overlap:
            raise AssertionError('--moe-unperm2-mem-optim-swap currently only can be used with --moe-fb-overlap')
        if args.moe_unperm2_mem_optim and args.moe_unperm2_mem_optim_swap:
            raise AssertionError('--moe-unperm2-mem-optim and --moe-unperm2-mem-optim-swap are incompatible.')


    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.mlp import mlp_init
        from mindspeed.core.pipeline_parallel.fb_overlap import (
            linear_backward_wgrad_detach,
            group_mlp_forward_detach,
            transformer_layer_forward_backward_overlaping,
            gpt_model_forward_backward_overlaping,
            forward_backward_pipelining_with_interleaving
        )
        from mindspeed.core.pipeline_parallel.fb_overlap.adaptor import _make_param_hook

        if getattr(args, self.feature_name, None):
            patch_manager.register_patch('megatron.core.transformer.mlp.MLP.__init__', mlp_init)
            patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.forward', gpt_model_forward_backward_overlaping)
            patch_manager.register_patch('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward_detach)
            patch_manager.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer.forward',
                                         transformer_layer_forward_backward_overlaping)
            patch_manager.register_patch('mindspeed.core.transformer.transformer_block.NoopTransformerLayer.forward',
                                         transformer_layer_forward_backward_overlaping)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
                                         linear_backward_wgrad_detach)
            patch_manager.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_param_hook',
                                         _make_param_hook)
            if getattr(args, 'num_layers_per_virtual_pipeline_stage', None):
                patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving',
                                             forward_backward_pipelining_with_interleaving)



