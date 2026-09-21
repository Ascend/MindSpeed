# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Own asynchronous forward sends while Megatron pseudo-deallocates its outputs."""

from contextvars import ContextVar
from functools import wraps

import torch

from mindspeed.args_utils import get_full_args

_ACTIVE_SENDS = ContextVar("mindspeed_deferred_p2p_sends", default=None)
# A failed device/communication wait cannot establish safe reclamation. Keep the
# affected storage alive until process exit, and propagate the failure to training.
_FAILED_SENDS = []


class _PendingSend:
    def __init__(self, tensor):
        # A Python reference to the original Tensor is insufficient: Megatron
        # replaces its .data. detach() gives us independent Tensor metadata while
        # retaining the original storage, without a payload copy or autograd graph.
        self.tensor = tensor.detach()
        self.work = None

    def finish(self):
        device = self.tensor.device
        if self.work is None:
            # Submission raised before returning a Work (possibly after enqueueing
            # a send). Only this exceptional path needs a device synchronization.
            torch.npu.synchronize(device)
        else:
            with torch.npu.device(device):
                stream = torch.npu.current_stream(device)
                if self.work.wait() is False:
                    raise RuntimeError("Deferred P2P send did not complete successfully")
                # HCCL Work.wait() may only enqueue an event dependency on the
                # current stream. Protect the storage past the HOST return, even
                # when it was originally allocated on another stream.
                self.tensor.record_stream(stream)
        self.tensor = None
        self.work = None


class _DeferredSends:
    def __init__(self):
        self.pending = []

    def begin(self, tensor):
        entry = _PendingSend(tensor)
        # Take ownership BEFORE native submission, including its exception path.
        self.pending.append(entry)
        return entry

    def retire_previous(self):
        # Submit the current send first, then wait for the previous one, restoring
        # the old interleaved schedule's ordering. Keep the current source alive.
        while len(self.pending) > 1:
            self.pending[0].finish()
            self.pending.pop(0)

    def close(self):
        try:
            while self.pending:
                self.pending[0].finish()
                self.pending.pop(0)
        except BaseException:
            _FAILED_SENDS.extend(self.pending)
            self.pending.clear()
            raise


def forward_backward_pipelining_with_interleaving_wrapper(schedule):
    """Drain owned sends on normal return, forward-only return and exceptions."""

    @wraps(schedule)
    def wrapper(*args, **kwargs):
        # Patches are installed before full argument validation. Read its result
        # once per schedule invocation, rather than on every microbatch/send.
        if not getattr(get_full_args(), "defer_p2p_send_wait", False):
            return schedule(*args, **kwargs)
        sends = _DeferredSends()
        token = _ACTIVE_SENDS.set(sends)
        try:
            return schedule(*args, **kwargs)
        finally:
            try:
                sends.close()
            finally:
                _ACTIVE_SENDS.reset(token)

    return wrapper


def send_forward_recv_forward_wrapper(send_forward_recv_forward):
    """Transfer ownership of send_next from the schedule to its scoped manager."""

    @wraps(send_forward_recv_forward)
    def wrapper(self, output_tensor, recv_prev, tensor_shape, overlap_p2p_comm=False):
        sends = _ACTIVE_SENDS.get()
        # Preserve short-circuit order before accessing optional tensor metadata.
        if (
            sends is None  # pylint: disable=too-many-boolean-expressions
            or not overlap_p2p_comm
            or output_tensor is None
            or not self.config.deallocate_pipeline_outputs
            or self.config.batch_p2p_comm
            or not isinstance(output_tensor, torch.Tensor)
            or output_tensor.device.type != "npu"
        ):
            return send_forward_recv_forward(self, output_tensor, recv_prev, tensor_shape, overlap_p2p_comm)
        entry = sends.begin(output_tensor)
        input_tensor, handles = send_forward_recv_forward(self, entry.tensor, recv_prev, tensor_shape, overlap_p2p_comm)
        if not isinstance(handles, dict) or handles.get("send_next") is None:
            raise RuntimeError("Deferred P2P send requires Megatron 0.18 send_next Work handles")
        entry.work = handles["send_next"]
        sends.retire_previous()
        # The schedule must still wait on all receives. Only the owned send is
        # removed, so Megatron's immediate pre-deallocation wait has no send Work.
        remaining_handles = dict(handles)
        del remaining_handles["send_next"]
        return input_tensor, remaining_handles

    return wrapper
