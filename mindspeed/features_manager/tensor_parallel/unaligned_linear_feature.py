from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class UnalignedLinearFeature(MindSpeedFeature):

    def __init__(self, feature_name):
        super().__init__(feature_name)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--unaligned-linear', action='store_true',
                           help='Replace ColumnParallelLinear/RowParallelLinear with '
                                'UnalignedColumnParallelLinearAdaptor/UnalignedRowParallelLinearAdaptor.')

    def validate_args(self, args):
        self.incompatible_check(args, 'use_ascend_mc2')
        # self.dependency_check(..)

    def register_patches(self, patch_manager, args):
        from mindspeed.core.tensor_parallel.unaligned_layers.adaptor import divide_adaptor, \
            scatter_to_sequence_parallel_region_adaptor, get_rotary_seq_len, UnalignedColumnParallelLinearAdaptor, \
            UnalignedRowParallelLinearAdaptor, reduce_scatter_to_sequence_parallel_region_adaptor
        if getattr(args, self.feature_name, None):
            patch_manager.register_patch('megatron.core.utils.divide', divide_adaptor)
            patch_manager.register_patch(
                'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len',
                get_rotary_seq_len)
            patch_manager.register_patch('megatron.core.tensor_parallel.mappings.scatter_to_sequence_parallel_region',
                                         scatter_to_sequence_parallel_region_adaptor)
            patch_manager.register_patch('megatron.core.tensor_parallel.mappings.reduce_scatter_to_sequence_parallel_region',
                                         reduce_scatter_to_sequence_parallel_region_adaptor)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear',
                                         UnalignedColumnParallelLinearAdaptor)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear',
                                         UnalignedRowParallelLinearAdaptor)
