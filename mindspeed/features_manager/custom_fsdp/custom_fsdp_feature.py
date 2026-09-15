# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
from mindspeed.features_manager.feature import MindSpeedFeature


class CustomFSDPFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('use-megatron-fsdp')

    def register_args(self, parser):
        # Also register with the adaptor's early parser, before MCore adds its
        # DistributedDataParallelConfig arguments to the training parser.
        for flag in ('--use-megatron-fsdp', '--use-custom-fsdp'):
            if not self._is_arg_registered(parser, flag):
                parser.add_argument(flag, action='store_true', help='Enable Megatron FSDP.')

    def is_need_apply(self, args):
        enabled = getattr(args, 'use_megatron_fsdp', False) or getattr(args, 'use_custom_fsdp', False)
        return enabled and self.optimization_level <= self._parse_optimization_level(
            getattr(args, 'optimization_level', 2)
        )

    def pre_validate_args(self, args):
        if getattr(args, 'use_custom_fsdp', False) or getattr(args, 'use_megatron_fsdp', False):
            # MCore 0.18 selects the model, optimizer and checkpoint paths using
            # use_megatron_fsdp. Normalize the legacy flag before its validation.
            args.use_megatron_fsdp = True
            args.use_custom_fsdp = True

    def register_patches(self, patch_manager, args):
        from mindspeed.core.distributed.custom_fsdp.param_and_grad_buffer import (
            bucket_group_gradient_reduce,
            gradient_reduce_preprocessing,
        )

        target = 'megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer'
        patch_manager.register_patch(f'{target}.gradient_reduce_preprocessing', gradient_reduce_preprocessing)
        patch_manager.register_patch(
            f'{target}.GradReducePipeline._bucket_group_gradient_reduce', bucket_group_gradient_reduce
        )
