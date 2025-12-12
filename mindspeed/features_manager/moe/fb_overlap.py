#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import warnings
from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class MoEFwdBwdOverlapFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('moe-fb-overlap')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--moe-fb-overlap', action='store_true')
        group.add_argument('--moe-unperm2-mem-optim-swap', action='store_true')

    def validate_args(self, args):
        self.incompatible_check(args, 'moe_alltoall_overlap_comm')
        self.incompatible_check(args, 'overlap_grad_reduce')
        self.incompatible_check(args, 'moe_hierarchical_alltoallv')
        self.incompatible_check(args, 'moe_zero_memory_num_layers')
        self.incompatible_check(args, 'use_nanopipe')
        self.incompatible_check(args, 'automated_pipeline')
        self.incompatible_check(args, 'recompute_in_bubble')
        self.incompatible_check(args, 'recompute_in_advance')
        self.incompatible_check(args, 'use_legacy_models')
        self.incompatible_check(args, 'moe_tp_extend_ep')
        self.incompatible_check(args, 'swap_attention')
        self.dependency_check(args, 'moe_grouped_gemm')
        if args.moe_fb_overlap and args.moe_token_dispatcher_type in ['allgather', 'alltoall_seq']:
            raise AssertionError('The fb overlap feature do not support allgather and alltoall_seq dispatcher.')

        if args.moe_fb_overlap and (args.expert_tensor_parallel_size != 1 or args.expert_model_parallel_size == 1):
            raise AssertionError(
                'fb overlap only support expert-tensor-parallel-size=1 and expert-model-parallel-size > 1')

        if args.moe_unperm2_mem_optim_swap and not args.moe_fb_overlap:
            raise AssertionError('--moe-unperm2-mem-optim-swap currently only can be used with --moe-fb-overlap')


        incorrect_schedule = (getattr(args, 'schedules_method', None) != 'dualpipev'
                            and getattr(args, 'num_layers_per_virtual_pipeline_stage', None) is None
                            and int(getattr(args, 'pipeline_model_parallel_size', 1)) != 1)
        if args.moe_fb_overlap and incorrect_schedule:
            raise AssertionError('The fb overlap needs no pipeline, virtual pipeline or dualpipeV schedules.')

        if getattr(args, 'virtual_pipeline_model_parallel_size', None) is not None and args.moe_fb_overlap:
            # In VPP schedule, do a GBS check.
            if not args.global_batch_size // (args.micro_batch_size * args.pipeline_model_parallel_size * args.data_parallel_size) > 1:
                raise ValueError(f"""In VPP schedule, 
                        fb_overlap needs global_batch_size // (micro_batch_size * pipeline_model_parallel_size * data_parallel_size) > 1.
                        The global_batch_size is {args.global_batch_size},
                        but the micro_batch_size is {args.micro_batch_size}, PP size is {args.pipeline_model_parallel_size},DP size is {args.data_parallel_size}.
                        """)

    def post_validate_args(self, args):
        # Noop check.
        if args.noop_layers is not None and args.moe_fb_overlap and getattr(args, 'schedules_method', None) != 'dualpipev':
            noop_layers_list = list(args.noop_layers)
            if noop_layers_list[0] < (args.num_layers - args.num_layers_per_virtual_pipeline_stage):
                raise AssertionError('In VPP schedule with fb_overlap, the noop-layers must in last VPP stage.')

    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            from mindspeed.core.transformer.moe.moe_feature.fb_overlap import (
                linear_backward_wgrad_detach,
                transformer_block_fb_overlap_init_wrapper,
                mtp_block_fb_overlap_forward_wrapper,
                dualpipev_fb_overlap_mtp_layer_forward
            )
            from mindspeed.core.transformer.moe.moe_feature.fb_overlap.adaptor import (
                _make_backward_post_hook,
                get_moe_module_spec_wrapper,
                get_forward_backward_func_vpp_overlap_wrapper
            )
            patch_manager.register_patch('megatron.core.models.gpt.moe_module_specs.get_moe_module_spec', get_moe_module_spec_wrapper)
            patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.__init__',
                                         transformer_block_fb_overlap_init_wrapper)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
                                         linear_backward_wgrad_detach)
            patch_manager.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_backward_post_hook',
                                         _make_backward_post_hook)

            if getattr(args, 'num_layers_per_virtual_pipeline_stage', None) or int(getattr(args, 'pipeline_model_parallel_size', 1)) == 1:
                patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                                              get_forward_backward_func_vpp_overlap_wrapper)

            if getattr(args, 'mtp_num_layers', None):
                patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.MultiTokenPredictionBlock.forward', 
                                             mtp_block_fb_overlap_forward_wrapper)
                patch_manager.register_patch('megatron.core.transformer.multi_token_prediction.MultiTokenPredictionLayer.forward', 
                                             dualpipev_fb_overlap_mtp_layer_forward)

