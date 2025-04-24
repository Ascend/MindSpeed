# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class TP2dFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('tp-2d')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--tp-2d', action='store_true', default=False,
                           help='use use-2d-tp to replace megatron-style tensor parallel')
        group.add_argument('--tp-x', type=int, default=1,
                           help='the fist dim tensor parallel size for Linear')
        group.add_argument('--tp-y', type=int, default=1,
                           help='the second dim tensor parallel size for Linear')
        group.add_argument('--enable-overlap-ag-with-matmul', action='store_true', default=False,
                           help='use enable-overlap-ag-with-matmul to overlap all-gather with matmul')
        group.add_argument('--enable-overlap-matmul-with-rs', action='store_true', default=False,
                           help='use enable-overlap-matmul-with-rs to overlap matmul with reduce-scatter')
        group.add_argument('--enable-backward-overlap-ag-with-matmul', action='store_true', default=False,
                           help='use enable-backward-overlap-ag-with-matmul to overlap all-gather with matmul in backward')

    def validate_args(self, args):
        self.incompatible_check(args, 'sequence_parallel')
        self.incompatible_check(args, 'use_fused_rmsnorm')
        self.incompatible_check(args, 'use_nanopipe')
        self.incompatible_check(args, 'use_ascend_coc')
        if getattr(args, self.feature_name, None):
            if getattr(args, 'context_parallel_algo', 'ulysses_cp_algo') not in ['megatron_cp_algo', 'ulysses_cp_algo']:
                raise AssertionError('tp-2d now only support megatron_cp_algo or ulysses_cp_algo')
            if args.tensor_model_parallel_size // args.tp_x != args.tp_y:
                raise AssertionError('need satisfy tp = tp_x * tp_y')
            if args.expert_model_parallel_size > 1:
                raise AssertionError('2d tp does not support moe')

    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            from mindspeed.core.tensor_parallel.tp_2d.norm_factory_2d import get_norm_tp_2d
            patch_manager.register_patch('megatron.legacy.model.utils.get_norm', get_norm_tp_2d)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_allreduce_layernorm_grads_wrapper
            patch_manager.register_patch('megatron.core.distributed.finalize_model_grads._allreduce_layernorm_grads',
                                 mindspeed_allreduce_layernorm_grads_wrapper)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_mlp_init_wrapper
            patch_manager.register_patch('megatron.core.transformer.mlp.MLP.__init__', mindspeed_mlp_init_wrapper)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_language_model_embedding_forward_wrapper
            patch_manager.register_patch('megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.forward',
                                 mindspeed_language_model_embedding_forward_wrapper)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_get_tensor_shapes_wrapper
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_tensor_shapes',
                                mindspeed_get_tensor_shapes_wrapper)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_forward_backward_pipelining_with_interleaving_tp2d
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving',
                                mindspeed_forward_backward_pipelining_with_interleaving_tp2d)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_transformer_block_forward_wrapper
            patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.forward', mindspeed_transformer_block_forward_wrapper)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_transformer_config_post_init
            patch_manager.register_patch('megatron.core.transformer.transformer_config.TransformerConfig.__post_init__',
                                mindspeed_transformer_config_post_init)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_initialize_model_parallel_wrapper
            patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel', mindspeed_initialize_model_parallel_wrapper)
            
            from mindspeed.core.tensor_parallel.tp_2d.adaptor import MindSpeedRotaryEmbedding2D
            patch_manager.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding',
                                    MindSpeedRotaryEmbedding2D)
            
            # mcore_transformer_adaptation(l2)
            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_attention_init, mindspeed_self_attention_init_wrapper
            patch_manager.register_patch('megatron.core.transformer.attention.SelfAttention.__init__', mindspeed_self_attention_init_wrapper)
            patch_manager.register_patch('megatron.core.transformer.attention.Attention.__init__', mindspeed_attention_init)
            patch_manager.register_patch('megatron.core.transformer.attention.SelfAttention.__init__', mindspeed_self_attention_init_wrapper)

        self.more_patches_for_tp2d(patch_manager, args)

    def more_patches_for_tp2d(self, patch_manager, args):
        """common features"""
        # mcore_transformer_adaptation(l2)
        from mindspeed.core.transformer.module import megatron_module_init_wrapper
        patch_manager.register_patch('megatron.core.transformer.module.MegatronModule.__init__', megatron_module_init_wrapper)

        from mindspeed.utils import get_batch_on_this_cp_rank
        patch_manager.register_patch('megatron.training.utils.get_batch_on_this_cp_rank', get_batch_on_this_cp_rank)
