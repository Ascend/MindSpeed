# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ContextParallelFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('context-parallel-size')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--context-parallel-algo', type=str, default='megatron_cp_algo',
                           choices=['megatron_cp_algo', 'hybrid_cp_algo', 'hybrid_adaptive_cp_algo'],
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
            if args.seq_length % (2 * args.context_parallel_size) != 0:
                raise AssertionError("sequence length must be divisible by 2 * context_parallel_size")
            if args.position_embedding_type == 'alibi':
                if not (args.alibi_fusion_attn_type in [2, 3] and args.attention_mask_type == 'causal'):
                    raise AssertionError("megatron_cp_algo only support alibi type in [2, 3] and attention_mask_type is causal")

            if not (args.cp_window_size >= 1 and args.cp_window_size < args.context_parallel_size):
                raise AssertionError('cp_window_size should in range [1, context_parallel_size) when using double_ring_attention.')
            n_window, remainder = divmod(args.context_parallel_size, args.cp_window_size)
            if not (n_window >= 1 and remainder == 0):
                raise AssertionError('context parallel size must be divisible by cp_window_size when using double ring attention.')
            args.use_flash_attn = True

        if args.context_parallel_size > 1 and args.position_embedding_type == 'alibi':
            if args.context_parallel_algo != 'megatron_cp_algo':
                raise AssertionError("alibi only support megatron_cp_algo")

        if args.context_parallel_size > 1 and args.reset_attention_mask and args.attention_mask_type == 'causal':
            if args.context_parallel_algo != 'megatron_cp_algo':
                raise AssertionError('accelerated eod reset mode only support ring attention')

        # hybrid context parallel
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'hybrid_cp_algo':
            if args.ulysses_degree_in_cp is None:
                raise AssertionError("--ulysses-degree-in-cp must be specified in hybrid_cp_algo")
            ring_degree, remainder = divmod(args.context_parallel_size, args.ulysses_degree_in_cp)
            if not (ring_degree > 1 and remainder == 0):
                raise AssertionError("--ulysses-degree-in-cp must be devisible by --context-parallel-size")
            args.ring_degree = ring_degree

            head, remainder = divmod(args.num_attention_heads,
                                     args.ulysses_degree_in_cp * args.tensor_model_parallel_size)
            if not (head >= 1 and remainder == 0):
                raise AssertionError("num_attention_heads must be divisible by ulysse-degree-in-cp * tensor_model_parallel_size in hybrid cp")

            if args.seq_length % (2 * args.context_parallel_size) != 0:
                raise AssertionError("sequence length must be divisible by 2 * context_parallel_size in hybrid cp")

            if not (args.cp_window_size >= 1 and args.cp_window_size < ring_degree):
                raise AssertionError('cp_window_size should be in range [1, ring_degree) when using double ring attention with hybrid context parallelism.')
            n_window, remainder = divmod(ring_degree, args.cp_window_size)
            if not (n_window >= 1 and remainder == 0):
                raise AssertionError('ring_degree should be divisible by cp_window_size when using double ring with hybrid context parallelism.')
            args.use_flash_attn = True


    def register_patches(self, patch_manager, args):
        if int(getattr(args, 'context_parallel_size', 1)) > 1:
            # The patches of eod
            if getattr(args, 'reset_attention_mask', False):
                from mindspeed.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids, collate_wrapper
                from mindspeed.utils import get_batch_on_this_cp_rank_wrapper
                patch_manager.register_patch('megatron.core.datasets.gpt_dataset._get_ltor_masks_and_position_ids',
                                    _get_ltor_masks_and_position_ids)
                patch_manager.register_patch('torch.utils.data._utils.collate.default_collate', collate_wrapper)
                patch_manager.register_patch('megatron.training.utils.get_batch_on_this_cp_rank', get_batch_on_this_cp_rank_wrapper)

                from mindspeed.core.pipeline_parallel.p2p_communication import _p2p_ops_eod
                patch_manager.register_patch('megatron.core.pipeline_parallel.p2p_communication._p2p_ops', _p2p_ops_eod)
                from mindspeed.core.models.gpt.gpt_model import gpt_forward_wrapper
                patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.forward', gpt_forward_wrapper)
                from mindspeed.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb_thd
                patch_manager.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding._apply_rotary_pos_emb_thd',
                                    apply_rotary_pos_emb_thd)
                from mindspeed.core.transformer.attention import attention_forward
                patch_manager.register_patch('megatron.core.transformer.attention.Attention.forward', attention_forward)

                from mindspeed.core.models.common.embeddings.rotary_pos_embedding import rotary_forward, Eod_get_rotary_seq_len
                patch_manager.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.forward',
                                    rotary_forward)
                patch_manager.register_patch(
                    'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len',
                    Eod_get_rotary_seq_len)

            from mindspeed.core.context_parallel.adaptor import MindSpeedCPDotProductAttention
            patch_manager.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention', MindSpeedCPDotProductAttention)

            from mindspeed.core.context_parallel.adaptor import attention_init_wrapper
            patch_manager.register_patch('megatron.core.transformer.attention.Attention.__init__', attention_init_wrapper)

            from mindspeed.core.context_parallel.model_parallel_utils import initialize_model_parallel_cp_wrapper, destroy_model_parallel_cp_wrapper, get_context_parallel_group_for_send_recv_overlap

            patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                                         initialize_model_parallel_cp_wrapper)
            patch_manager.register_patch('megatron.core.parallel_state.destroy_model_parallel',
                                         destroy_model_parallel_cp_wrapper)
            patch_manager.register_patch('megatron.core.parallel_state.get_context_parallel_group_for_send_recv_overlap',
                                         get_context_parallel_group_for_send_recv_overlap)

            from mindspeed.core.context_parallel.get_batch_utils import get_batch_on_this_cp_rank, get_batch_on_this_tp_rank
            patch_manager.register_patch('megatron.training.utils.get_batch_on_this_cp_rank', get_batch_on_this_cp_rank)
            patch_manager.register_patch('megatron.training.utils.get_batch_on_this_tp_rank', get_batch_on_this_tp_rank)

            from mindspeed.core.context_parallel.rotary_pos_embedding_utils import get_pos_emb_on_this_cp_rank
            patch_manager.register_patch(
                'megatron.core.models.common.embeddings.rotary_pos_embedding.get_pos_emb_on_this_cp_rank',
                get_pos_emb_on_this_cp_rank)
