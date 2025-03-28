#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class DualpipeVFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('schedules-method')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--schedules-method', type=str,
                           default=None, choices=['dualpipev'])

    def validate_args(self, args):
        if args.schedules_method == "dualpipev":
            if args.num_layers_per_virtual_pipeline_stage is not None:
                raise AssertionError(
                    "The dualpipev and virtual_pipeline are incompatible.")
            if args.num_layers < args.pipeline_model_parallel_size * 2:
                raise AssertionError(
                    'number of layers must be at least 2*pipeline_model_parallel_size in dualpipe')
            num_micro_batch = args.global_batch_size // args.micro_batch_size // args.data_parallel_size
            if num_micro_batch < args.pipeline_model_parallel_size * 2 - 1:
                raise AssertionError(
                    "num_micro_batch should more than pipeline_model_parallel_size * 2 - 1")

    def register_patches(self, patch_manager, args):
        from megatron.training.utils import print_rank_0
        from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import forward_backward_pipelining_with_cutinhalf
        from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_chunks import get_model, dualpipev_fp16forward, get_num_layers_to_build, train_step, finalize_model_grads

        if args.schedules_method == "dualpipev":

            patch_manager.register_patch(
                'megatron.training.training.get_model', get_model)
            patch_manager.register_patch(
                'megatron.training.training.train_step', train_step)
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving',
                                         forward_backward_pipelining_with_cutinhalf)
            patch_manager.register_patch(
                'megatron.legacy.model.module.Float16Module.forward', dualpipev_fp16forward)
            patch_manager.register_patch(
                'megatron.core.transformer.transformer_block.get_num_layers_to_build', get_num_layers_to_build)
            patch_manager.register_patch(
                'megatron.training.utils.print_rank_last', print_rank_0)
            patch_manager.register_patch(
                'megatron.core.distributed.finalize_model_grads.finalize_model_grads', finalize_model_grads)
