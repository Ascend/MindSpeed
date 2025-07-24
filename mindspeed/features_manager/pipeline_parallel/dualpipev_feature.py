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
        group.add_argument('--dualpipev-dw-detach', action='store_true',
                           help='detach dw in cooldown to reduce bubble')

    def validate_args(self, args):
        if args.schedules_method == "dualpipev":
            if args.overlap_grad_reduce:
                raise AssertionError(
                    "The dualpipev and overlap_grad_reduce are incompatible.")
            if not args.untie_embeddings_and_output_weights:
                raise AssertionError(
                    "The dualpipev requires untie_embeddings_and_output_weights.")
            if args.swap_attention:
                raise AssertionError(
                    "The dualpipev and swap_attention are incompatible.")
            if args.context_parallel_size > 1:
                raise AssertionError(
                    "The dualpipev and context_parallel are incompatible.")
            if args.num_layers_per_virtual_pipeline_stage is not None:
                raise AssertionError(
                    "The dualpipev and virtual_pipeline are incompatible.")
            if args.pipeline_model_parallel_size == 1:
                raise AssertionError(
                    "pipeline_model_parallel_size should be larger than 1 with dualpipev schedules")
            if args.num_layers < args.pipeline_model_parallel_size * 2:
                raise AssertionError(
                    'number of layers must be at least 2*pipeline_model_parallel_size in dualpipe')
            num_micro_batch = args.global_batch_size // args.micro_batch_size // args.data_parallel_size
            if num_micro_batch < args.pipeline_model_parallel_size * 2 - 1:
                raise AssertionError(
                    "num_micro_batch should more than pipeline_model_parallel_size * 2 - 1")
            if args.tp_2d:
                raise AssertionError(
                    "The dualpipev and tp_2d are incompatible.")

    def register_patches(self, patch_manager, args):
        from megatron.training.utils import print_rank_0
        from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import forward_backward_pipelining_with_cutinhalf
        from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_chunks import (
            get_model, dualpipev_fp16forward, get_num_layers_to_build, train_step,
            _allreduce_embedding_grads_wrapper, evaluate, get_transformer_layer_offset, pretrain, get_batch_on_this_tp_rank
        )
        from mindspeed.core.pipeline_parallel.dualpipev.mtp_utils import (setup_embeddings_and_output_layer_with_mtp,
                                                                         dualpipev_get_mtp_num_layers_to_build)

        if args.schedules_method == "dualpipev":

            patch_manager.register_patch(
                'megatron.training.training.get_model', get_model)
            patch_manager.register_patch(
                'megatron.training.training.train_step', train_step)
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving',
                                         forward_backward_pipelining_with_cutinhalf)
            patch_manager.register_patch(
                'megatron.core.transformer.module.Float16Module.forward', dualpipev_fp16forward)
            patch_manager.register_patch(
                'megatron.core.transformer.transformer_block.get_num_layers_to_build', get_num_layers_to_build)
            patch_manager.register_patch(
                'megatron.training.utils.print_rank_last', print_rank_0)
            patch_manager.register_patch(
                'megatron.core.distributed.finalize_model_grads._allreduce_embedding_grads', _allreduce_embedding_grads_wrapper)
            patch_manager.register_patch('megatron.training.training.evaluate', evaluate)
            patch_manager.register_patch('megatron.core.transformer.transformer_layer.get_transformer_layer_offset', get_transformer_layer_offset)
            patch_manager.register_patch('megatron.training.training.pretrain', pretrain)
            patch_manager.register_patch('megatron.training.utils.get_batch_on_this_tp_rank', get_batch_on_this_tp_rank)

            if getattr(args, 'mtp_num_layers', None):
                patch_manager.register_patch("megatron.core.models.common.language_module.language_module.LanguageModule.setup_embeddings_and_output_layer", 
                                            setup_embeddings_and_output_layer_with_mtp)
                patch_manager.register_patch("megatron.core.transformer.multi_token_prediction.get_mtp_num_layers_to_build",
                                            dualpipev_get_mtp_num_layers_to_build)
