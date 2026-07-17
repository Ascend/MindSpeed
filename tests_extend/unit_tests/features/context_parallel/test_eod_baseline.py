from argparse import Namespace

import torch
import pytest

from mindspeed.core.context_parallel.get_batch_utils import set_actual_seq_len
from mindspeed.core.transformer.flash_attention.reset_attention_mask import adaptor
from mindspeed.core.transformer.flash_attention.reset_attention_mask import utils
from mindspeed.features_manager.transformer.flash_attention.reset_attention_mask_feature import (
    ResetAttentionMaskFeature,
)
from mindspeed.utils import get_position_ids, set_position_ids


def test_eod_batch_patches_follow_megatron_018_core_utils_entrypoints():
    class PatchRecorder:
        def __init__(self):
            self.targets = []

        def register_patch(self, target, *args, **kwargs):
            self.targets.append(target)

    patch_manager = PatchRecorder()
    ResetAttentionMaskFeature().register_patches(
        patch_manager,
        Namespace(reset_attention_mask=True),
    )

    assert 'megatron.core.utils.get_batch_on_this_tp_rank' in patch_manager.targets
    assert 'megatron.core.utils.get_batch_on_this_cp_rank' in patch_manager.targets
    assert not any(target.startswith('megatron.training.utils.get_batch_on_this_') for target in patch_manager.targets)


def test_fix_sub_sequence_collate_offsets_every_micro_batch_sample(monkeypatch):
    monkeypatch.setattr(utils, 'get_args', lambda: Namespace(seq_length=4, fix_sub_seq_length=2))

    samples = [
        {
            'tokens': torch.tensor([1, 2, 3, 4]),
            'position_ids': (torch.arange(4), torch.tensor([4])),
        },
        {
            'tokens': torch.tensor([5, 6, 7, 8]),
            'position_ids': (torch.arange(4), torch.tensor([4])),
        },
    ]

    def collate(items):
        return {key: torch.stack([item[key] for item in items]) for key in items[0]}

    batch = utils.collate_wrapper(collate)(samples)

    assert torch.equal(batch['actual_seq_len'], torch.tensor([2, 4, 6, 8]))


def test_p2p_eod_uses_document_aware_ring_layout(monkeypatch):
    args = Namespace(
        reset_attention_mask=True,
        attention_mask_type='causal',
        context_parallel_size=2,
        context_parallel_algo='megatron_cp_algo',
    )
    monkeypatch.setattr(utils, 'get_args', lambda: args)
    monkeypatch.setattr(utils.mpu, 'get_context_parallel_world_size', lambda: 2)
    monkeypatch.setattr(utils.mpu, 'get_context_parallel_rank', lambda: 0)
    monkeypatch.setattr(utils, 'get_runtime_actual_seq_len', lambda: torch.tensor([4, 8]))
    calls = []

    def megatron_slicer(batch, cp_group=None):
        calls.append(cp_group)
        return batch

    batch = {
        'tokens': torch.arange(16).reshape(1, 16),
        'position_ids': torch.arange(16).reshape(1, 16),
    }
    cp_group = object()

    result = utils.get_batch_on_this_cp_rank_wrapper(megatron_slicer)(batch, cp_group)

    assert result is batch
    assert calls == []
    assert torch.equal(result['tokens'], torch.tensor([[0, 1, 6, 7, 8, 9, 14, 15]]))


def test_all_gather_eod_keeps_megatron_as_the_only_batch_slice_owner(monkeypatch):
    args = Namespace(
        reset_attention_mask=True,
        attention_mask_type='causal',
        context_parallel_size=2,
        context_parallel_algo='kvallgather_cp_algo',
    )
    monkeypatch.setattr(utils, 'get_args', lambda: args)
    calls = []

    def megatron_slicer(batch, cp_group=None):
        calls.append(cp_group)
        return batch

    batch = {'position_ids': torch.tensor([[0, 1, 2, 3]])}
    cp_group = object()
    result = utils.get_batch_on_this_cp_rank_wrapper(megatron_slicer)(batch, cp_group)

    assert result is batch
    assert calls == [cp_group]


def test_attention_wrapper_folds_only_inside_attention_and_restores_batch():
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    position_ids = torch.tensor([[0, 0], [1, 1], [2, 2]])
    set_position_ids(position_ids)
    packed_seq_params = Namespace(qkv_format='thd')

    def megatron_attention(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        packed_seq_params=None,
        position_ids=None,
    ):
        assert hidden_states.shape == (6, 1, 2)
        assert torch.equal(get_position_ids().squeeze(1), torch.tensor([0, 1, 2, 0, 1, 2]))
        return hidden_states + 1, None

    output, _ = adaptor.attention_forward_wrapper(megatron_attention)(
        None,
        hidden_states,
        None,
        packed_seq_params=packed_seq_params,
    )

    assert output.shape == hidden_states.shape
    assert torch.equal(output, hidden_states + 1)
    assert torch.equal(get_position_ids(), position_ids)


def test_gpt_wrapper_keeps_micro_batch_shape():
    input_ids = torch.arange(6).reshape(2, 3)
    position_ids = torch.tensor([[0, 1, 0], [0, 1, 2]])
    set_actual_seq_len(torch.tensor([2, 3, 6]))
    def megatron_gpt(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        packed_seq_params=None,
    ):
        assert input_ids.shape == (2, 3)
        assert position_ids.shape == (2, 3)
        assert packed_seq_params.qkv_format == 'thd'
        return input_ids

    result = adaptor.gpt_forward_wrapper(megatron_gpt)(
        None,
        input_ids,
        position_ids,
        None,
    )

    assert torch.equal(result, input_ids)


def test_p2p_eod_rope_length_uses_document_position_range():
    set_position_ids(torch.tensor([[0, 1], [3070, 3071]], dtype=torch.int64))
    packed_seq_params = Namespace(qkv_format='thd')

    def native_get_rotary_seq_len(
        self,
        inference_context,
        transformer,
        transformer_input,
        transformer_config,
        packed_seq_params=None,
    ):
        return 2

    result = adaptor.rotary_seq_len_eod_wrapper(native_get_rotary_seq_len)(
        None, None, None, None, None, packed_seq_params
    )

    assert result == 3072


def test_sequence_parallel_multi_batch_is_rejected_before_token_reordering():
    args = Namespace(
        reset_attention_mask=True,
        context_parallel_size=1,
        attention_mask_type='causal',
        sequence_parallel=True,
        micro_batch_size=2,
        sft=False,
        apply_rope_fusion=False,
    )

    with pytest.raises(AssertionError, match='sequence parallel currently requires'):
        ResetAttentionMaskFeature().validate_args(args)


def test_cp_eod_multi_batch_is_rejected_before_collective_reordering():
    args = Namespace(
        reset_attention_mask=True,
        context_parallel_size=2,
        attention_mask_type='general',
        sequence_parallel=False,
        micro_batch_size=2,
        sft=False,
        apply_rope_fusion=False,
    )

    with pytest.raises(AssertionError, match='CP with reset-attention-mask currently requires'):
        ResetAttentionMaskFeature().validate_args(args)
