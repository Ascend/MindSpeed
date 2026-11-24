# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
