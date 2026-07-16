from argparse import Namespace

import pytest

from mindspeed.core.context_parallel.model_parallel_utils import (
    TENPU_NATIVE,
    UNSUPPORTED_CP,
    get_cp_backend_route,
)
from mindspeed.core.context_parallel.tenpu_adaptor import set_tenpu_cp_runtime_options
from mindspeed.features_manager.context_parallel.context_parallel_feature import ContextParallelFeature


def _make_args(**overrides):
    args = Namespace(
        context_parallel_size=2,
        context_parallel_algo=None,
        cp_comm_type=['p2p'],
        reset_attention_mask=True,
        attention_mask_type='general',
        transformer_impl='transformer_engine',
        seq_length=128,
        cp_window_size=1,
        use_cp_send_recv_overlap=False,
        use_fused_ring_attention_update=False,
        position_embedding_type='rope',
        alibi_fusion_attn_type=None,
        use_flash_attn=False,
        tp_2d=False,
        tp_y=1,
        ulysses_degree_in_cp=None,
        hierarchical_context_parallel_sizes=None,
        num_attention_heads=8,
        tensor_model_parallel_size=1,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_p2p_eod_general_is_skipped_without_mindspeed_ring():
    feature = ContextParallelFeature()
    args = _make_args()

    feature.pre_validate_args(args)
    assert args.attention_mask_type == 'general'
    assert get_cp_backend_route(args) == UNSUPPORTED_CP
    with pytest.raises(AssertionError, match='Unsupported CP capability cell'):
        feature.validate_args(args)


def test_p2p_eod_causal_uses_tenpu():
    args = _make_args(attention_mask_type='causal')

    assert get_cp_backend_route(args) == TENPU_NATIVE
    ContextParallelFeature().validate_args(args)


@pytest.mark.parametrize('mask_type', ['causal', 'general'])
def test_a2a_eod_is_explicitly_unsupported(mask_type):
    feature = ContextParallelFeature()
    args = _make_args(cp_comm_type=['a2a'], attention_mask_type=mask_type)

    feature.pre_validate_args(args)

    assert args.context_parallel_algo == 'ulysses_cp_algo'
    assert get_cp_backend_route(args) == UNSUPPORTED_CP
    with pytest.raises(AssertionError, match='Unsupported CP capability cell'):
        feature.validate_args(args)


def test_all_gather_eod_general_is_explicitly_unsupported():
    args = _make_args(
        context_parallel_algo='kvallgather_cp_algo',
        cp_comm_type=['all_gather'],
    )

    assert get_cp_backend_route(args) == UNSUPPORTED_CP
    with pytest.raises(AssertionError, match='no MindSpeed EOD-only fallback'):
        ContextParallelFeature().validate_args(args)


def test_non_eod_hierarchical_cp_is_owned_by_tenpu():
    args = _make_args(
        reset_attention_mask=False,
        context_parallel_algo='hybrid_cp_algo',
        cp_comm_type=['a2a+p2p'],
        hierarchical_context_parallel_sizes=[2, 1],
    )

    assert get_cp_backend_route(args) == TENPU_NATIVE
    ContextParallelFeature().validate_args(args)


def test_unknown_non_eod_cp_type_does_not_fall_through_to_tenpu():
    args = _make_args(reset_attention_mask=False, cp_comm_type=['unknown'])

    assert get_cp_backend_route(args) == UNSUPPORTED_CP


@pytest.mark.parametrize(
    'retired_type',
    ['megatron_cp_algo', 'ulysses_cp_algo', 'kvallgather_cp_algo', 'hybrid_cp_algo'],
)
def test_retired_mindspeed_algo_is_not_accepted_as_cp_comm_type(retired_type):
    args = _make_args(cp_comm_type=[retired_type])

    assert get_cp_backend_route(args) == UNSUPPORTED_CP
    with pytest.raises(AssertionError, match='Unsupported CP capability cell'):
        ContextParallelFeature().validate_args(args)


def test_megatron_allgather_spelling_is_normalized_for_tenpu():
    args = _make_args(cp_comm_type=['allgather'], attention_mask_type='causal')

    assert get_cp_backend_route(args) == TENPU_NATIVE


def test_eod_rejects_heterogeneous_per_layer_cp_comm_type():
    args = _make_args(cp_comm_type=['p2p', 'a2a'])

    with pytest.raises(AssertionError, match='homogeneous'):
        ContextParallelFeature().validate_args(args)


def test_non_eod_heterogeneous_general_is_owned_by_tenpu():
    args = _make_args(
        reset_attention_mask=False,
        cp_comm_type=['p2p', 'allgather'],
        attention_mask_type='general',
    )

    assert get_cp_backend_route(args) == TENPU_NATIVE
    ContextParallelFeature().validate_args(args)


def test_non_eod_heterogeneous_unknown_cp_type_is_rejected():
    args = _make_args(reset_attention_mask=False, cp_comm_type=['p2p', 'unknown'])

    assert get_cp_backend_route(args) == UNSUPPORTED_CP


def test_legacy_context_parallel_algo_overrides_megatron_default_cp_comm_type():
    args = _make_args(context_parallel_algo='ulysses_cp_algo', cp_comm_type=['p2p'])

    ContextParallelFeature().pre_validate_args(args)

    assert args.context_parallel_algo == 'ulysses_cp_algo'
    assert args.cp_comm_type == ['a2a']


def test_p2p_overlap_is_allowed_and_uses_tenpu_runtime_mapping():
    args = _make_args(attention_mask_type='causal', use_cp_send_recv_overlap=True)

    ContextParallelFeature().validate_args(args)

    class Attention:
        pass

    attention = Attention()
    set_tenpu_cp_runtime_options(attention, args)
    assert attention.cp_window_size == 1
    assert attention.use_cp_send_recv_overlap is True
    assert attention.ulysses_degree_in_cp is None


def test_p2p_custom_window_is_allowed_when_it_partitions_cp():
    args = _make_args(
        context_parallel_size=4,
        attention_mask_type='causal',
        cp_window_size=2,
    )

    ContextParallelFeature().validate_args(args)


def test_window_and_overlap_are_rejected_for_non_p2p_cp():
    args = _make_args(
        reset_attention_mask=False,
        cp_comm_type=['all_gather'],
        cp_window_size=2,
    )

    with pytest.raises(AssertionError, match='only with --cp-comm-type p2p or a2a\\+p2p'):
        ContextParallelFeature().validate_args(args)


def test_hierarchical_cp_window_uses_p2p_subgroup_size():
    args = _make_args(
        context_parallel_size=4,
        reset_attention_mask=False,
        context_parallel_algo='hybrid_cp_algo',
        cp_comm_type=['a2a+p2p'],
        hierarchical_context_parallel_sizes=[2, 2],
        cp_window_size=1,
        use_cp_send_recv_overlap=True,
    )

    ContextParallelFeature().validate_args(args)

    attention = type('Attention', (), {})()
    set_tenpu_cp_runtime_options(attention, args)
    assert attention.ulysses_degree_in_cp == 2


def test_context_parallel_feature_uses_early_string_values_without_becoming_default_patch():
    class PatchManager:
        def __init__(self):
            self.calls = []

        def register_patch(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    patch_manager = PatchManager()
    ContextParallelFeature().register_patches(
        patch_manager,
        _make_args(context_parallel_size='1', cp_window_size='1', attention_mask_type='causal'),
    )

    assert patch_manager.calls == []

    ContextParallelFeature().register_patches(
        patch_manager,
        _make_args(context_parallel_size='4', cp_window_size='2', attention_mask_type='causal'),
    )

    assert len(patch_manager.calls) == 1
    assert patch_manager.calls[0][0][0] == (
        'megatron.core.extensions.transformer_engine.TEDotProductAttention.__init__'
    )
