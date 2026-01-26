# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import is_megatron_training_available


class ContextParallelFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('context-parallel-size')
        
    def is_need_apply(self, args):
        """Check the feature is need to apply."""
        return (self.optimization_level <= args.optimization_level and getattr(args, self.feature_name, 1)) \
            or self.default_patches

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--context-parallel-algo', type=str, default='megatron_cp_algo',
                           choices=['megatron_cp_algo', 'hybrid_cp_algo', 'hybrid_adaptive_cp_algo', 'kvallgather_cp_algo'],
                           help='context parallel algorithm')

        # ring context parallel
        group.add_argument('--cp-window-size', type=int, default=1)
        group.add_argument('--attention-mask-type', type=str, default='causal',
                           choices=['causal', 'general'], help='context parallel attention mask type')
        group.add_argument('--use-cp-send-recv-overlap', action='store_true',
                           help='use this flag to enable cp send-recv-overlap.')
        group.add_argument("--use-fused-ring-attention-update", action='store_true',
                           help="Use fused ring attention update.")
        group.add_argument("--megatron-cp-in-bnsd", action='store_true',
                           help="Megatron CP in bnsd.")


    def validate_args(self, args):
        # ring context parallel
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'megatron_cp_algo':
            if hasattr(args, 'seq_length') and args.seq_length % (2 * args.context_parallel_size) != 0:
                raise AssertionError("sequence length must be divisible by 2 * context_parallel_size")
            if getattr(args, 'position_embedding_type', None) == 'alibi':
                if not ((args.alibi_fusion_attn_type == 2) and (args.attention_mask_type == 'causal')):
                    raise AssertionError("megatron_cp_algo only support alibi type 2 and attention_mask_type causal")

            if not (args.cp_window_size >= 1 and args.cp_window_size < args.context_parallel_size):
                raise AssertionError('cp_window_size should in range [1, context_parallel_size) when using double_ring_attention.')
            n_window, remainder = divmod(args.context_parallel_size, args.cp_window_size)
            if not (n_window >= 1 and remainder == 0):
                raise AssertionError('context parallel size must be divisible by cp_window_size when using double ring attention.')
            args.use_flash_attn = True

        if args.context_parallel_size > 1 and getattr(args, 'position_embedding_type', None) == 'alibi':
            if args.context_parallel_algo != 'megatron_cp_algo':
                raise AssertionError("alibi only support megatron_cp_algo")

        # hybrid context parallel
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'hybrid_cp_algo':
            if args.ulysses_degree_in_cp is None:
                raise AssertionError("--ulysses-degree-in-cp must be specified in hybrid_cp_algo")
            ring_degree, remainder = divmod(args.context_parallel_size, args.ulysses_degree_in_cp)
            if not (ring_degree > 1 and remainder == 0):
                raise AssertionError("--ulysses-degree-in-cp must be divisible by --context-parallel-size and "
                                     "--ulysses-degree-in-cp divided by --context-parallel-size must be greater than 1")
            args.ring_degree = ring_degree

            head, remainder = divmod(args.num_attention_heads,
                                     args.ulysses_degree_in_cp * args.tensor_model_parallel_size)
            if not (head >= 1 and remainder == 0):
                raise AssertionError("num_attention_heads must be divisible by ulysse-degree-in-cp * tensor_model_parallel_size in hybrid cp")

            if hasattr(args, 'seq_length') and args.seq_length % (2 * args.context_parallel_size) != 0:
                raise AssertionError("sequence length must be divisible by 2 * context_parallel_size in hybrid cp")

            if not (args.cp_window_size >= 1 and args.cp_window_size < ring_degree):
                raise AssertionError('cp_window_size should be in range [1, ring_degree) when using double ring attention with hybrid context parallelism.')
            n_window, remainder = divmod(ring_degree, args.cp_window_size)
            if not (n_window >= 1 and remainder == 0):
                raise AssertionError('ring_degree should be divisible by cp_window_size when using double ring with hybrid context parallelism.')
            args.use_flash_attn = True

        # kvallgather context parallel
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'kvallgather_cp_algo':
            if args.attention_mask_type != "causal":
                raise AssertionError("kvallgather_cp_algo only supports causal attention mask type")
                
            if not getattr(args, 'reset_attention_mask', False):
                if hasattr(args, 'seq_length') and args.seq_length % (2 * args.context_parallel_size) != 0:
                    raise AssertionError("sequence length must be divisible by 2 * context_parallel_size in kvallgather_cp_algo with SBHD format")
            else:
                if hasattr(args, 'seq_length') and args.seq_length % args.context_parallel_size != 0:
                    raise AssertionError("sequence length must be divisible by context_parallel_size in kvallgather_cp_algo with THD format")

    def register_patches(self, patch_manager, args):
        _use_cp = int(getattr(args, 'context_parallel_size', 1)) > 1
        _cp_algo = getattr(args, 'context_parallel_algo', 'megatron_cp_algo')
        _cp_expanded_by_2d_tp = getattr(args, 'tp_2d', False) and getattr(args, 'tp_y', 1) > 1

        if _use_cp or (_cp_expanded_by_2d_tp and _cp_algo == 'megatron_cp_algo'):
            from mindspeed.core.context_parallel.adaptor import MindSpeedCPDotProductAttention
            from mindspeed.te.pytorch.attention.dot_product_attention.dot_product_attention import MindSpeedTEDotProductAttention

            patch_manager.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention',
                                         MindSpeedCPDotProductAttention)
            if _cp_algo == 'kvallgather_cp_algo':
                patch_manager.register_patch('megatron.core.extensions.transformer_engine.TEDotProductAttention',
                                             MindSpeedTEDotProductAttention)
            else:
                patch_manager.register_patch('megatron.core.extensions.transformer_engine.TEDotProductAttention',
                                             MindSpeedCPDotProductAttention)

            from mindspeed.core.context_parallel.adaptor import attention_init_wrapper
            from mindspeed.core.context_parallel.adaptor import multi_latent_attention_init_wrapper
            patch_manager.register_patch('megatron.core.transformer.attention.Attention.__init__', attention_init_wrapper)
            patch_manager.register_patch('megatron.core.transformer.multi_latent_attention.MultiLatentAttention.__init__',
                                         multi_latent_attention_init_wrapper)

            from mindspeed.core.context_parallel.model_parallel_utils import (
                initialize_model_parallel_cp_wrapper,
                destroy_model_parallel_cp_wrapper,
                get_context_parallel_group_for_send_recv_overlap
            )
            patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                                         initialize_model_parallel_cp_wrapper)
            patch_manager.register_patch('megatron.core.parallel_state.destroy_model_parallel',
                                         destroy_model_parallel_cp_wrapper)
            patch_manager.register_patch('megatron.core.parallel_state.get_context_parallel_group_for_send_recv_overlap',
                                         get_context_parallel_group_for_send_recv_overlap)

            megatron_training_available = is_megatron_training_available()
            if megatron_training_available:
                from mindspeed.core.context_parallel.get_batch_utils import get_batch_on_this_cp_rank
                patch_manager.register_patch('megatron.training.utils.get_batch_on_this_cp_rank', get_batch_on_this_cp_rank)

            from mindspeed.core.context_parallel.rotary_pos_embedding_utils import get_pos_emb_on_this_cp_rank
            patch_manager.register_patch(
                'megatron.core.models.common.embeddings.rotary_pos_embedding.get_pos_emb_on_this_cp_rank',
                get_pos_emb_on_this_cp_rank)