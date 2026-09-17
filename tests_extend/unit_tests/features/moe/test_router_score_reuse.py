# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import (
    compute_routing_scores_for_aux_loss,
    topk_routing_with_score_function,
)
from mindspeed.core.transformer.moe.moe_utils import (
    compute_routing_scores_for_aux_loss_wrapper,
    get_fixed_router_indices,
    topk_routing_with_score_function_wrapper,
)
from mindspeed.core.transformer.moe.router_score_reuse import (
    router_score_reuse_wrapper,
    topk_routing_with_aux_score_reuse,
)
from mindspeed.features_manager.moe.router_score_reuse import MoERouterScoreReuseFeature


ROUTER_CASES = (
    pytest.param("softmax", None, None, False, id="softmax"),
    pytest.param("softmax", 2, 1, False, id="softmax-grouped"),
    pytest.param("sigmoid", None, None, False, id="sigmoid"),
    pytest.param("sigmoid", None, None, True, id="sigmoid-bias"),
    pytest.param("sigmoid", 2, 1, False, id="sigmoid-grouped"),
)


@pytest.mark.parametrize(
    "optimization_level,enabled,expected",
    (
        pytest.param(0, True, False, id="level-0-disabled"),
        pytest.param(1, True, False, id="level-1-disabled"),
        pytest.param(2, False, False, id="feature-disabled"),
        pytest.param(2, True, True, id="feature-enabled"),
    ),
)
def test_router_score_reuse_feature_is_parameter_controlled(optimization_level, enabled, expected):
    feature = MoERouterScoreReuseFeature()
    args = SimpleNamespace(
        optimization_level=optimization_level,
        moe_router_score_reuse=enabled,
    )

    assert feature.is_need_apply(args) is expected


def test_router_score_reuse_argument_defaults_to_disabled():
    feature = MoERouterScoreReuseFeature()
    parser = ArgumentParser()

    feature.register_args(parser)

    assert not parser.parse_args([]).moe_router_score_reuse
    assert parser.parse_args(["--moe-router-score-reuse"]).moe_router_score_reuse


def test_router_score_reuse_feature_registers_topk_router_patch():
    feature = MoERouterScoreReuseFeature()
    patch_manager = mock.Mock()

    feature.register_patches(patch_manager, SimpleNamespace())

    patch_manager.register_patch.assert_called_once_with(
        "megatron.core.transformer.moe.router.TopKRouter.routing",
        router_score_reuse_wrapper,
    )


def _assert_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def _make_inputs(with_bias, dtype=torch.float32):
    torch.manual_seed(1234)
    logits = torch.randn(8, 4, dtype=dtype, requires_grad=True)
    expert_bias = torch.randn(4, dtype=torch.float32) if with_bias else None
    padding_mask = torch.tensor([False, True, False, False, True, False, False, False])
    return logits, expert_bias, padding_mask


def _run_reference(logits, score_function, num_groups, group_topk, expert_bias, padding_mask, fixed_router):
    dispatch = topk_routing_with_score_function
    auxiliary = compute_routing_scores_for_aux_loss
    if fixed_router:
        dispatch = topk_routing_with_score_function_wrapper(dispatch)
        auxiliary = compute_routing_scores_for_aux_loss_wrapper(auxiliary)

    probs, routing_map = dispatch(
        logits,
        2,
        use_pre_softmax=score_function == "softmax",
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=0.7,
        score_function=score_function,
        expert_bias=expert_bias,
    )
    routing_map_for_aux_loss, scores_for_aux_loss = auxiliary(
        logits,
        2,
        score_function,
        padding_mask=padding_mask,
    )
    return probs, routing_map, routing_map_for_aux_loss, scores_for_aux_loss


@pytest.mark.parametrize("score_function,num_groups,group_topk,with_bias", ROUTER_CASES)
@pytest.mark.parametrize("fixed_router", (False, True), ids=("regular", "fixed"))
@pytest.mark.parametrize("router_dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
def test_router_score_reuse_is_strictly_aligned(
    score_function, num_groups, group_topk, with_bias, fixed_router, router_dtype
):
    reference_logits, expert_bias, padding_mask = _make_inputs(with_bias, dtype=router_dtype)
    reused_logits = reference_logits.detach().clone().requires_grad_(True)

    reference_outputs = _run_reference(
        reference_logits,
        score_function,
        num_groups,
        group_topk,
        expert_bias,
        padding_mask,
        fixed_router,
    )
    reused_outputs = topk_routing_with_aux_score_reuse(
        reused_logits,
        2,
        use_pre_softmax=score_function == "softmax",
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=0.7,
        score_function=score_function,
        expert_bias=expert_bias,
        padding_mask=padding_mask,
        fixed_router=fixed_router,
    )

    for reused, reference in zip(reused_outputs, reference_outputs):
        _assert_equal(reused, reference)

    torch.manual_seed(4321)
    probs_weight = torch.randn_like(reference_outputs[0])
    scores_weight = torch.randn_like(reference_outputs[3])
    reference_loss = (reference_outputs[0] * probs_weight).sum() + (
        reference_outputs[3] * scores_weight
    ).sum()
    reused_loss = (reused_outputs[0] * probs_weight).sum() + (
        reused_outputs[3] * scores_weight
    ).sum()
    _assert_equal(reused_loss, reference_loss)

    reference_loss.backward()
    reused_loss.backward()
    _assert_equal(reused_logits.grad, reference_logits.grad)


def _count_score_and_topk_ops(score_function, fixed_router, router_dtype):
    logits, _, padding_mask = _make_inputs(with_bias=False, dtype=router_dtype)
    original_softmax = torch.softmax
    original_sigmoid = torch.sigmoid
    original_topk = torch.topk
    with (
        mock.patch.object(torch, "softmax", wraps=original_softmax) as softmax_mock,
        mock.patch.object(torch, "sigmoid", wraps=original_sigmoid) as sigmoid_mock,
        mock.patch.object(torch, "topk", wraps=original_topk) as topk_mock,
    ):
        topk_routing_with_aux_score_reuse(
            logits,
            2,
            use_pre_softmax=score_function == "softmax",
            score_function=score_function,
            padding_mask=padding_mask,
            fixed_router=fixed_router,
        )
    return softmax_mock.call_count, sigmoid_mock.call_count, topk_mock.call_count


@pytest.mark.parametrize(
    "score_function,expected_counts",
    (
        pytest.param("softmax", (1, 0, 1), id="softmax-one-score-one-topk"),
        pytest.param("sigmoid", (0, 1, 2), id="sigmoid-one-score-two-topks"),
    ),
)
def test_fix_router_only_changes_assignment(score_function, expected_counts):
    for router_dtype in (torch.float32, torch.bfloat16):
        regular_counts = _count_score_and_topk_ops(
            score_function, fixed_router=False, router_dtype=router_dtype
        )
        fixed_counts = _count_score_and_topk_ops(
            score_function, fixed_router=True, router_dtype=router_dtype
        )

        assert regular_counts == expected_counts
        assert fixed_counts == regular_counts


@pytest.mark.parametrize("fixed_router", (False, True), ids=("regular", "fixed"))
def test_bf16_router_wrapper_uses_reuse_path(fixed_router):
    original_routing = mock.Mock(side_effect=AssertionError("unexpected Megatron fallback"))
    wrapped_routing = router_score_reuse_wrapper(original_routing)
    router = mock.Mock()
    router.training = True
    router.is_aux_loss_enabled.return_value = True
    router.routing_type = "aux_loss"
    router.score_function = "softmax"
    router.topk = 2
    router.expert_bias = None
    router.router_replay = None
    router.config = SimpleNamespace(
        num_moe_experts=4,
        moe_router_fusion=False,
        moe_router_pre_softmax=True,
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_router_topk_scaling_factor=0.7,
        moe_expert_capacity_factor=None,
    )
    router.apply_z_loss.side_effect = lambda logits, **kwargs: logits
    router._apply_aux_loss.side_effect = lambda probs, *args, **kwargs: probs
    router._apply_seq_aux_loss.side_effect = lambda probs, *args, **kwargs: probs
    router._apply_global_aux_loss.side_effect = lambda probs, *args, **kwargs: probs

    torch.manual_seed(1234)
    logits = torch.randn(2, 4, 4, dtype=torch.bfloat16, requires_grad=True)
    runtime_args = SimpleNamespace(moe_tp_extend_ep=False, fix_router=fixed_router)
    original_softmax = torch.softmax
    original_topk = torch.topk
    with (
        mock.patch(
            "mindspeed.core.transformer.moe.router_score_reuse._get_runtime_args",
            return_value=runtime_args,
        ),
        mock.patch.object(torch, "softmax", wraps=original_softmax) as softmax_mock,
        mock.patch.object(torch, "topk", wraps=original_topk) as topk_mock,
    ):
        wrapped_routing(router, logits)

    original_routing.assert_not_called()
    assert softmax_mock.call_count == 1
    assert topk_mock.call_count == 1


@pytest.mark.parametrize("score_function", ("softmax", "sigmoid", "sqrtsoftplus"))
def test_fix_router_aux_fallback_replaces_only_assignment(score_function):
    reference_logits, _, padding_mask = _make_inputs(with_bias=False)
    fixed_logits = reference_logits.detach().clone().requires_grad_(True)

    _, reference_scores = compute_routing_scores_for_aux_loss(
        reference_logits,
        2,
        score_function,
        padding_mask=padding_mask,
    )
    fixed_auxiliary = compute_routing_scores_for_aux_loss_wrapper(
        compute_routing_scores_for_aux_loss
    )
    original_topk = torch.topk
    with mock.patch.object(torch, "topk", wraps=original_topk) as topk_mock:
        fixed_map, fixed_scores = fixed_auxiliary(
            fixed_logits,
            2,
            score_function,
            padding_mask=padding_mask,
        )

    fixed_indices = get_fixed_router_indices(fixed_logits, 2)
    expected_map = torch.zeros_like(fixed_logits).int().scatter(1, fixed_indices, 1).bool()
    expected_map = expected_map * (~padding_mask).unsqueeze(-1)

    _assert_equal(fixed_scores, reference_scores)
    _assert_equal(fixed_map, expected_map)
    assert topk_mock.call_count == 1

    torch.manual_seed(5678)
    scores_weight = torch.randn_like(reference_scores)
    reference_loss = (reference_scores * scores_weight).sum()
    fixed_loss = (fixed_scores * scores_weight).sum()
    reference_loss.backward()
    fixed_loss.backward()
    _assert_equal(fixed_logits.grad, reference_logits.grad)
