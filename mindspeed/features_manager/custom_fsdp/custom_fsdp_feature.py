# Copyright 2025-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        flag = '--fsdp-memory-compatibility'
        if not self._is_arg_registered(parser, flag):
            parser.add_argument(
                flag,
                action='store_true',
                help='Run Megatron FSDP parameter all-gather on the calling stream '
                'and disable extra prefetch during start_param_sync.',
            )

    def is_need_apply(self, args):
        enabled = getattr(args, 'use_megatron_fsdp', False) or getattr(args, 'use_custom_fsdp', False)
        return enabled and self.optimization_level <= self._parse_optimization_level(
            getattr(args, 'optimization_level', 2)
        )

    def pre_validate_args(self, args):
        if getattr(args, 'fsdp_memory_compatibility', False) and not self.is_need_apply(args):
            raise ValueError(
                '--fsdp-memory-compatibility requires --use-megatron-fsdp '
                '(or --use-custom-fsdp) and optimization level >= 2.'
            )
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
        if getattr(args, 'fsdp_memory_compatibility', False):
            from mindspeed.core.distributed.custom_fsdp.memory_compatibility import (
                all_gather_pipeline_init_on_current_stream_wrapper,
                all_gather_ready_no_extra_prefetch_wrapper,
                start_param_sync_no_extra_prefetch_wrapper,
            )

            patch_manager.register_patch(
                f'{target}.AllGatherPipeline.__init__',
                all_gather_pipeline_init_on_current_stream_wrapper,
            )
            fsdp = 'megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp.MegatronFSDP'
            patch_manager.register_patch(f'{fsdp}.start_param_sync', start_param_sync_no_extra_prefetch_wrapper)
            patch_manager.register_patch(
                f'{fsdp}.all_gather_and_wait_parameters_ready', all_gather_ready_no_extra_prefetch_wrapper
            )
