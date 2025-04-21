# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class RiPipeSchedulesFeature(MindSpeedFeature):

    def validate_args(self, args):
        if getattr(args, "adaptive_recompute_device_size", None) is not None:
            adaptive_recompute_enable = args.adaptive_recompute_device_size > 0 or args.adaptive_recompute_device_swap
            if adaptive_recompute_enable:
                if args.recompute_in_advance or args.recompute_in_bubble:
                    raise AssertionError('adaptive selective recompute is not compatible with ripipe schedule.')
        self.incompatible_check(args, "optimize_send_recv_comm")
        if (getattr(args, "ampipe_degree", None) is not None and args.ampipe_degree > 1
                and getattr(args, self.feature_name, None)):
            raise AssertionError('{} and {} are incompatible.'.format(self.feature_name, "ampipe"))

    def register_patches(self, patch_manager, args):
        if args.recompute_in_bubble is None and args.recompute_in_advance is None:
            return
        from mindspeed.core.pipeline_parallel.ripipe_schedules import get_forward_backward_func_ripipe_patch
        patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                                     get_forward_backward_func_ripipe_patch)


class RiPipeSchedulesBubbleFeature(RiPipeSchedulesFeature):

    def __init__(self):
        super().__init__("recompute-in-bubble")

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--recompute-in-bubble", action="store_true",
                           help="use bubble to do recompute to reduce memory.")

    def validate_args(self, args):
        super().validate_args(args)
        self.incompatible_check(args, "adaptive_memory_optimization")
        if args.recompute_in_bubble:
            if args.recompute_num_layers:
                raise AssertionError('recompute_num_layers must be None or 0 when using recompute_in_bubble')
            if args.pipeline_model_parallel_size <= 1 or args.num_layers_per_virtual_pipeline_stage is None:
                raise AssertionError('recompute_in_bubble only support pipelining with interleaving')
            if getattr(args, "swap_attention", None) is None:
                # Following is a trick to realize bubble recomputation. We first enable all recomputation,
                # and then disable recomputation for all layers except the ones chosen for bubble recomputation.
                args.recompute_granularity = "full"
                args.recompute_method = "block"
            if getattr(args, "enable_recompute_layers_per_pp_rank", None) is not None:
                args.recompute_num_layers = args.num_layers // args.pipeline_model_parallel_size
            else:
                args.recompute_num_layers = args.num_layers_per_virtual_pipeline_stage


class RiPipeSchedulesAdvanceFeature(RiPipeSchedulesFeature):

    def __init__(self):
        super().__init__("recompute-in-advance")

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--recompute-in-advance', action='store_true',
                           help='recompute early to reduce bubble and improve training.')

    def validate_args(self, args):
        super().validate_args(args)
        args.reduce_recompute_for_last_chunk = False
        if args.recompute_in_advance:
            args.reduce_recompute_for_last_chunk = True
            if args.recompute_method == "uniform":
                raise AssertionError('recompute_in_advance does not support uniform recompute_method')
            if not args.recompute_num_layers and not getattr(args, "adaptive_memory_optimization", None):
                raise AssertionError('recompute_num_layers can not be None or 0 when using recompute_in_advance')
            if args.pipeline_model_parallel_size <= 1 or args.num_layers_per_virtual_pipeline_stage is None:
                raise AssertionError('recompute_in_advance only support pipelining with interleaving')
            if args.num_layers_per_virtual_pipeline_stage != 1:
                args.recompute_in_advance = False
