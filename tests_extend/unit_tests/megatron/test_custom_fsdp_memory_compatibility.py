# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CPU contract tests for FSDP memory compatibility; no training runtime needed."""

from types import SimpleNamespace

import pytest

from mindspeed.core.distributed.custom_fsdp import memory_compatibility


_FORWARD_ORDER = object()


def _instance(**config):
    return SimpleNamespace(ddp_config=SimpleNamespace(overlap_param_gather=True, **config))


@pytest.fixture(autouse=True)
def sync_context():
    context = memory_compatibility._START_SYNC_INSTANCE
    token = context.set(None)
    try:
        yield context
    finally:
        context.reset(token)


@pytest.fixture
def gather_env():
    env = SimpleNamespace(calls=[], result=object(), error=None)

    # Match the MCore 0.18 gather signature without importing MCore or torch.
    def original(instance, params, prefetch=True, prefetch_order=_FORWARD_ORDER,
                 wait_bucket_ready=True, bwd=False):
        env.calls.append((instance, params, prefetch, prefetch_order, wait_bucket_ready, bwd))
        if env.error is not None:
            raise env.error
        return env.result

    env.gather = memory_compatibility.all_gather_ready_no_extra_prefetch_wrapper(original)
    return env


def _assert_gather_call(call, instance, params, prefetch, order=_FORWARD_ORDER, wait=True, bwd=False):
    assert call[0] is instance
    assert call[1] is params
    assert call[2] is prefetch
    assert call[3] is order
    assert call[4] is wait
    assert call[5] is bwd


@pytest.mark.parametrize('scope', ['matching', 'other', 'after-start'])
@pytest.mark.parametrize('style', ['positional', 'keyword', 'mixed'])
@pytest.mark.parametrize('prefetch', [True, False], ids=['prefetch', 'no-prefetch'])
@pytest.mark.parametrize('bwd', [False, True], ids=['forward', 'backward'])
def test_gather_prefetch_is_scoped_to_matching_forward(
    gather_env, sync_context, scope, style, prefetch, bwd,
):
    owner, other = _instance(), _instance()
    assert owner == other and owner is not other
    target = other if scope == 'other' else owner
    params, order, start_result = [object()], object(), object()
    start_calls = []

    def invoke_gather():
        if style == 'positional':
            result = gather_env.gather(target, params, prefetch, order, False, bwd)
        elif style == 'keyword':
            result = gather_env.gather(
                target, params=params, prefetch=prefetch, prefetch_order=order,
                wait_bucket_ready=False, bwd=bwd,
            )
        else:
            result = gather_env.gather(
                target, params, prefetch, prefetch_order=order, wait_bucket_ready=False, bwd=bwd,
            )
        assert result is gather_env.result

    def original_start(instance, *unused, force_sync=False, force_dispatch=False):
        start_calls.append(instance)
        if scope != 'after-start':
            invoke_gather()
        return start_result

    start = memory_compatibility.start_param_sync_no_extra_prefetch_wrapper(original_start)
    assert start(owner) is start_result
    assert len(start_calls) == 1 and start_calls[0] is owner
    assert sync_context.get() is None
    if scope == 'after-start':
        invoke_gather()

    expected = False if scope == 'matching' and not bwd else prefetch
    assert len(gather_env.calls) == 1
    _assert_gather_call(gather_env.calls[0], target, params, expected, order, False, bwd)


@pytest.mark.parametrize('scope', ['matching', 'other', 'after-start'])
def test_gather_omitted_arguments_keep_upstream_defaults(gather_env, sync_context, scope):
    owner, other, params = _instance(), _instance(), [object()]
    target = other if scope == 'other' else owner

    def original_start(instance, *unused, force_sync=False, force_dispatch=False):
        if scope != 'after-start':
            return gather_env.gather(target, params)
        return gather_env.result

    start = memory_compatibility.start_param_sync_no_extra_prefetch_wrapper(original_start)
    assert start(owner) is gather_env.result
    assert sync_context.get() is None
    if scope == 'after-start':
        assert gather_env.gather(target, params) is gather_env.result
    assert len(gather_env.calls) == 1
    _assert_gather_call(gather_env.calls[0], target, params, scope != 'matching')


@pytest.mark.parametrize(
    'overlap,start_gather,expected_prefetch',
    [
        pytest.param(True, True, False, id='overlap-start-enabled'),
        pytest.param(True, False, True, id='overlap-start-disabled'),
        pytest.param(True, None, False, id='overlap-start-missing'),
        pytest.param(False, True, True, id='no-overlap-start-enabled'),
        pytest.param(False, False, True, id='no-overlap-start-disabled'),
        pytest.param(False, None, True, id='no-overlap-start-missing'),
    ],
)
def test_start_prefetch_config(gather_env, sync_context, overlap, start_gather, expected_prefetch):
    owner, params = _instance(), [object()]
    owner.ddp_config.overlap_param_gather = overlap
    if start_gather is not None:
        owner.ddp_config.fsdp_all_gather_in_start_param_sync = start_gather
    start_calls = []

    def original_start(instance, *unused, force_sync=False, force_dispatch=False):
        start_calls.append(instance)
        return gather_env.gather(instance, params)

    start = memory_compatibility.start_param_sync_no_extra_prefetch_wrapper(original_start)
    assert start(owner) is gather_env.result
    assert len(start_calls) == 1 and start_calls[0] is owner
    assert len(gather_env.calls) == 1
    _assert_gather_call(gather_env.calls[0], owner, params, expected_prefetch)
    assert sync_context.get() is None


@pytest.mark.parametrize(
    'unused,kwargs,expected_prefetch',
    [
        pytest.param((), {}, False, id='default-force-sync'),
        pytest.param((), {'force_sync': False}, False, id='explicit-async'),
        pytest.param((), {'force_sync': True}, True, id='forced-sync'),
        # In MCore 0.18 a positional True belongs to *unused, not force_sync.
        pytest.param((True,), {}, False, id='positional-true-is-unused'),
        pytest.param((True, 'unused'), {'force_dispatch': True}, False, id='force-dispatch'),
        pytest.param(
            (True, 'unused'), {'force_sync': True, 'force_dispatch': True}, True,
            id='forced-sync-and-dispatch',
        ),
    ],
)
def test_start_preserves_arguments(gather_env, sync_context, unused, kwargs, expected_prefetch):
    owner, params = _instance(), [object()]
    start_calls = []

    def original_start(instance, *unused, force_sync=False, force_dispatch=False):
        start_calls.append((instance, unused, force_sync, force_dispatch))
        return gather_env.gather(instance, params)

    start = memory_compatibility.start_param_sync_no_extra_prefetch_wrapper(original_start)
    assert start(owner, *unused, **kwargs) is gather_env.result
    assert len(start_calls) == 1
    assert start_calls[0][0] is owner
    assert start_calls[0][1:] == (
        unused, kwargs.get('force_sync', False), kwargs.get('force_dispatch', False),
    )
    assert len(gather_env.calls) == 1
    _assert_gather_call(gather_env.calls[0], owner, params, expected_prefetch)
    assert sync_context.get() is None


@pytest.mark.parametrize('outcome', ['return', 'start-error', 'gather-error'])
@pytest.mark.parametrize('prior_context', ['empty', 'existing'])
def test_start_restores_context(gather_env, sync_context, outcome, prior_context):
    owner, params = _instance(), [object()]
    previous = _instance() if prior_context == 'existing' else None
    token = sync_context.set(previous)
    error = RuntimeError(outcome)
    start_calls = []

    def original_start(instance, *unused, force_sync=False, force_dispatch=False):
        start_calls.append(instance)
        assert sync_context.get() is owner
        result = gather_env.gather(instance, params)
        if outcome == 'start-error':
            raise error
        return result

    start = memory_compatibility.start_param_sync_no_extra_prefetch_wrapper(original_start)
    if outcome == 'gather-error':
        gather_env.error = error
    try:
        if outcome == 'return':
            assert start(owner) is gather_env.result
        else:
            with pytest.raises(RuntimeError) as caught:
                start(owner)
            assert caught.value is error
        # Assert restoration before fixture/token cleanup can hide a leaked value.
        assert sync_context.get() is previous
        assert len(start_calls) == 1 and start_calls[0] is owner
        assert len(gather_env.calls) == 1
        _assert_gather_call(gather_env.calls[0], owner, params, False)
        gather_env.error = None
        assert gather_env.gather(owner, params) is gather_env.result
        assert len(gather_env.calls) == 2
        _assert_gather_call(gather_env.calls[1], owner, params, True)
        assert sync_context.get() is previous
    finally:
        sync_context.reset(token)


@pytest.mark.parametrize('outcome', ['return', 'raise'])
@pytest.mark.parametrize('inner_force_sync', [False, True], ids=['async-inner', 'sync-inner'])
def test_nested_start_restores_outer_instance(gather_env, sync_context, outcome, inner_force_sync):
    outer, inner, params = _instance(), _instance(), [object()]
    error, inner_result, outer_result = RuntimeError('inner start'), object(), object()
    start_calls = []

    def original_inner(instance, *unused, force_sync=False, force_dispatch=False):
        start_calls.append(instance)
        assert sync_context.get() is (None if force_sync else inner)
        assert gather_env.gather(inner, params) is gather_env.result
        assert gather_env.gather(outer, params) is gather_env.result
        if outcome == 'raise':
            raise error
        return inner_result

    inner_start = memory_compatibility.start_param_sync_no_extra_prefetch_wrapper(original_inner)

    def original_outer(instance, *unused, force_sync=False, force_dispatch=False):
        start_calls.append(instance)
        assert sync_context.get() is outer
        if outcome == 'raise':
            with pytest.raises(RuntimeError) as caught:
                inner_start(inner, force_sync=inner_force_sync)
            assert caught.value is error
        else:
            assert inner_start(inner, force_sync=inner_force_sync) is inner_result
        assert sync_context.get() is outer
        assert gather_env.gather(outer, params) is gather_env.result
        return outer_result

    outer_start = memory_compatibility.start_param_sync_no_extra_prefetch_wrapper(original_outer)
    assert outer_start(outer) is outer_result
    assert len(start_calls) == 2
    assert start_calls[0] is outer and start_calls[1] is inner
    assert sync_context.get() is None
    assert gather_env.gather(inner, params) is gather_env.result
    assert len(gather_env.calls) == 4
    _assert_gather_call(gather_env.calls[0], inner, params, inner_force_sync)
    _assert_gather_call(gather_env.calls[1], outer, params, True)
    _assert_gather_call(gather_env.calls[2], outer, params, False)
    _assert_gather_call(gather_env.calls[3], inner, params, True)


@pytest.mark.parametrize(
    'style', ['omitted', 'none-positional', 'none-keyword', 'stream-positional',
              'stream-keyword', 'buffer-keyword'],
)
def test_all_gather_init_always_receives_none_stream(style):
    instance, buffer, stream, result = object(), object(), object(), object()
    calls = []

    def original_init(self, param_and_grad_buffer, ag_stream=None):
        calls.append((self, param_and_grad_buffer, ag_stream))
        return result

    init = memory_compatibility.all_gather_pipeline_init_on_current_stream_wrapper(original_init)
    if style == 'omitted':
        actual = init(instance, buffer)
    elif style == 'none-positional':
        actual = init(instance, buffer, None)
    elif style == 'none-keyword':
        actual = init(instance, buffer, ag_stream=None)
    elif style == 'stream-positional':
        actual = init(instance, buffer, stream)
    elif style == 'stream-keyword':
        actual = init(instance, buffer, ag_stream=stream)
    else:
        actual = init(instance, param_and_grad_buffer=buffer, ag_stream=stream)
    assert actual is result
    assert len(calls) == 1
    assert calls[0][0] is instance
    assert calls[0][1] is buffer
    assert calls[0][2] is None
