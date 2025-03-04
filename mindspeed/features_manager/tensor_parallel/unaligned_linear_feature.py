from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class UnalignedLinearFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('unaligned-linear')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--unaligned-linear', action='store_true',
                           help='Replace ColumnParallelLinear/RowParallelLinear with '
                                'UnalignedColumnParallelLinearAdaptor/UnalignedRowParallelLinearAdaptor.')

    def validate_args(self, args):
        self.incompatible_check(args, 'use_ascend_mc2')
        self.incompatible_check(args, 'tp_2d')
        if args.unaligned_linear and args.num_experts and args.num_experts > 1:
            raise AssertionError("The unaligned linear feature does not support the moe model.")
        # self.dependency_check(..)

    def register_patches(self, patch_manager, args):
        from mindspeed.core.tensor_parallel.unaligned_layers.adaptor import divide_adaptor, \
            scatter_to_sequence_parallel_region_adaptor, get_rotary_seq_len, UnalignedColumnParallelLinearAdaptor, \
            UnalignedRowParallelLinearAdaptor, reduce_scatter_to_sequence_parallel_region_adaptor, \
            gather_from_sequence_parallel_region_adaptor
        from mindspeed.core.transformer.transformer_config import transformer_config_post_init
        from mindspeed.core.transformer.dot_product_attention import dot_product_attention_init_wrapper
        from mindspeed.core.transformer.attention import attention_init_wrapper
        if getattr(args, self.feature_name, None):
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear',
                                         UnalignedColumnParallelLinearAdaptor)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear',
                                         UnalignedRowParallelLinearAdaptor)

            # To adapt to the distribution of MHA attention heads
            patch_manager.register_patch('megatron.core.utils.divide', divide_adaptor)
            patch_manager.register_patch(
                'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len',
                get_rotary_seq_len)
            patch_manager.register_patch('megatron.core.transformer.transformer_config.TransformerConfig.__post_init__',
                                         transformer_config_post_init)

            # To adapt to the distribution of GQA attention heads
            patch_manager.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.__init__',
                                         dot_product_attention_init_wrapper)
            patch_manager.register_patch('megatron.core.transformer.attention.Attention.__init__',
                                         attention_init_wrapper)
            patch_manager.register_patch('megatron.core.tensor_parallel.mappings.gather_from_sequence_parallel_region',
                                         gather_from_sequence_parallel_region_adaptor)

            # To adapt to the sequence parallel feature
            patch_manager.register_patch('megatron.core.tensor_parallel.mappings.scatter_to_sequence_parallel_region',
                                         scatter_to_sequence_parallel_region_adaptor)
            patch_manager.register_patch('megatron.core.tensor_parallel.mappings.reduce_scatter_to_sequence_parallel_region',
                                         reduce_scatter_to_sequence_parallel_region_adaptor)