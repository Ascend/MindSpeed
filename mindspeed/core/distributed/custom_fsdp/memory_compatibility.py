# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Opt-in current-stream AG and start-sync prefetch compatibility for MCore 0.18."""

from contextvars import ContextVar
from functools import wraps
import inspect


_START_SYNC_INSTANCE = ContextVar('mindspeed_fsdp_start_sync_instance', default=None)


def all_gather_pipeline_init_on_current_stream_wrapper(original_init):
    """Let upstream select the calling stream on each gather, including after init."""

    @wraps(original_init)
    def init(self, param_and_grad_buffer, ag_stream=None):
        return original_init(self, param_and_grad_buffer, ag_stream=None)

    return init


def start_param_sync_no_extra_prefetch_wrapper(original):
    """Scope the legacy first-bucket prefetch boundary to this FSDP instance."""
    signature = inspect.signature(original)

    @wraps(original)
    def start(instance, *args, **kwargs):
        bound = signature.bind(instance, *args, **kwargs)
        bound.apply_defaults()
        config = instance.ddp_config
        suppress_prefetch = (
            not bound.arguments.get('force_sync', False)
            and config.overlap_param_gather
            and getattr(config, 'fsdp_all_gather_in_start_param_sync', True)
        )
        token = _START_SYNC_INSTANCE.set(instance if suppress_prefetch else None)
        try:
            return original(instance, *args, **kwargs)
        finally:
            _START_SYNC_INSTANCE.reset(token)

    return start


def all_gather_ready_no_extra_prefetch_wrapper(original):
    """Disable extra forward prefetch only within the matching start-sync call."""
    signature = inspect.signature(original)

    @wraps(original)
    def gather(instance, *args, **kwargs):
        if _START_SYNC_INSTANCE.get() is not instance:
            return original(instance, *args, **kwargs)
        bound = signature.bind(instance, *args, **kwargs)
        bound.apply_defaults()
        if not bound.arguments.get('bwd', False):
            bound.arguments['prefetch'] = False
        return original(*bound.args, **bound.kwargs)

    return gather
