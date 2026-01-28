# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ResetAttentionMaskFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('reset-attention-mask', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--fix-sub-seq-length', type=int, default=-1,
                           help='Sub-sequence length to process.  - If > 0: Specifies fixed sub-sequence length. '
                                'Each sub-sequence will have this length,  except the last one which will take the '
                                'remaining tokens. - If <= 0 or > seq_length: sub-sequences maintain their original'
                                ' lengths.')

    def validate_args(self, args):
        if args.context_parallel_size > 1 and hasattr(args, 'reset_attention_mask') and args.reset_attention_mask:
            if args.attention_mask_type == 'causal' and not getattr(args, 'variable_seq_lengths', None):
                raise AssertionError('accelerated eod reset mode needs variable_seq_lengths.')
    
    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            from mindspeed.core.transformer.flash_attention.reset_attention_mask.utils import (
                _get_ltor_masks_and_position_ids, collate_wrapper, get_batch_on_this_cp_rank_wrapper,
                eod_gptdataset_getitem)
            from mindspeed.core.transformer.flash_attention.reset_attention_mask.adaptor import (
                _p2p_ops_eod, gpt_forward_wrapper, attention_forward, MindSpeedMLASelfAttention,
                apply_rotary_pos_emb_thd, rotary_forward, Eod_get_rotary_seq_len)
            from mindspeed.core.context_parallel.get_batch_utils import get_batch_on_this_tp_rank

            patch_manager.register_patch('megatron.core.datasets.gpt_dataset.GPTDataset.__getitem__',
                                         eod_gptdataset_getitem)
            patch_manager.register_patch('megatron.core.datasets.gpt_dataset._get_ltor_masks_and_position_ids',
                                         _get_ltor_masks_and_position_ids)
            patch_manager.register_patch('torch.utils.data._utils.collate.default_collate', collate_wrapper)

            patch_manager.register_patch('megatron.training.utils.get_batch_on_this_tp_rank', get_batch_on_this_tp_rank)
            patch_manager.register_patch('megatron.training.utils.get_batch_on_this_cp_rank',
                                         get_batch_on_this_cp_rank_wrapper)

            patch_manager.register_patch('megatron.core.pipeline_parallel.p2p_communication._p2p_ops', _p2p_ops_eod)
            patch_manager.register_patch('megatron.core.pipeline_parallel.p2p_communication._batched_p2p_ops', _p2p_ops_eod)
            patch_manager.register_patch('megatron.core.models.gpt.gpt_model.GPTModel.forward', gpt_forward_wrapper)

            patch_manager.register_patch('megatron.core.transformer.attention.Attention.forward', attention_forward)
            if getattr(args, 'multi_latent_attention', None):
                patch_manager.register_patch(
                    'megatron.core.transformer.multi_latent_attention.MLASelfAttention',
                    MindSpeedMLASelfAttention)

            patch_manager.register_patch(
                'megatron.core.models.common.embeddings.rotary_pos_embedding._apply_rotary_pos_emb_thd',
                apply_rotary_pos_emb_thd)

            patch_manager.register_patch(
                'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.forward', rotary_forward)
            patch_manager.register_patch(
                'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len',
                Eod_get_rotary_seq_len)
