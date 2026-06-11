# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import functools
import logging
import os

from ..constants import WorkerStatus

logger = logging.getLogger(__name__)


def _send_stopped_heartbeat_and_exit(processor, exit_code=0):
    """Send STOPPED heartbeat then exit the process.

    Reports STOPPED status to Controller via Processor before calling os._exit(),
    so the Controller confirms the Processor has started exiting.
    """
    try:
        if processor is not None:
            if (
                hasattr(processor, 'heartbeat_manager')
                and processor.heartbeat_manager
                and processor.heartbeat_manager.running
            ):
                processor.heartbeat_manager.update_status(WorkerStatus.STOPPED)
                if hasattr(processor, '_send_heartbeat_now'):
                    processor._send_heartbeat_now()
    except Exception:
        logger.warning("[TTP] Failed to send stopped heartbeat before exit", exc_info=True)
    os._exit(exit_code)


def _handle_pause_and_dump(processor, timeout_action=300.0, timeout_dump=1800.0):
    """Handle PAUSE/DUMP/EXIT flow, consolidating repeated logic.

    Matches the proven stable reference implementation (de8fa3fd):
    wait_next_action serialises the main thread through _enter_wait_mode,
    then Controller DUMP_REQUEST (via dump_request_event) → daemon save
    thread → wait_for_dump_complete.

    No _paused_event or explicit PAUSE status update here — the main
    thread is already paused by the time wait_next_action is entered
    (the exception that brought us here interrupted NPU work).
    _enter_wait_mode (inside wait_next_action) handles the PAUSE status
    update.
    """
    action = processor.wait_next_action(timeout=timeout_action)

    if action == "EXIT":
        _send_stopped_heartbeat_and_exit(processor)
    elif action == "TIMEOUT":
        raise RuntimeError("wait_next_action timeout")

    if action == "DUMP":
        dump_success = processor.wait_for_dump_complete(timeout=timeout_dump)

        if not dump_success:
            raise RuntimeError("Dump failed")

        # Wait for the Controller to send EXIT confirmation after dump
        processor.wait_next_action(timeout=timeout_action)
        _send_stopped_heartbeat_and_exit(processor)

    # Fallback for unrecognized actions
    logger.warning("Unrecognized action '%s' from Controller, exiting", action)
    _send_stopped_heartbeat_and_exit(processor)


def ttp_exception_handler(func):
    """
    TTP exception handler decorator.

    Wraps Worker methods with exception handling and blocking logic.

    Usage:
        @ttp_exception_handler
        def update_actor(self, *args, **kwargs):
            ...
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        from .._registry import get_processor

        processor = get_processor()

        try:
            # Check if PAUSE is needed at method entry before execution.
            # If the Controller has sent a PAUSE notification, raise immediately.
            # Must be INSIDE try so RuntimeError("STEP FINISH") is caught below.
            if processor:
                processor.check_and_raise_if_paused()

            # Execute the original method
            result = func(self, *args, **kwargs)
            return result
        except RuntimeError as e:
            error_str = str(e)
            logger.error("[EXCEPTION_HANDLER] Caught RuntimeError: %s", error_str)

            # Handle STEP FINISH exception: PyThreadState_SetAsyncExc may inject
            # exceptions with no message, so also check _pause_type.
            is_step_finish = (error_str == "STEP FINISH") or (
                processor is not None and processor._pause_type == 'RAISE'
            )

            if is_step_finish and processor is not None:
                _handle_pause_and_dump(processor)

            # Handle legacy PAUSE exception
            if error_str == "PAUSE" and processor is not None:
                _handle_pause_and_dump(processor)

            # Check if this is a TTP-handled error.
            # FORCE STOP is caused by our own stop_device force-killing
            # in-flight NPU ops on ranks that missed the PAUSE signal.
            # It is NOT a new fault — the original fault rank already
            # reported.  Handle it separately: just enter PAUSE silently.
            if "FORCE STOP" in error_str and processor is not None:
                _handle_pause_and_dump(processor)

            is_ttp_error = any(
                s in error_str
                for s in [
                    "UCE",
                    "HBM",
                    "HCCL",
                    "NCCL",
                    "RuntimeError",
                    "connection",
                    "timeout",
                    "broken",
                    "TTP_INJECTED_FAULT",
                ]
            )

            if is_ttp_error and processor is not None:
                # Report the fault to the Controller so it can coordinate
                # the dump across all ranks.
                try:
                    processor.on_worker_exception(e)
                except Exception:
                    logger.warning("[TTP] Failed to report exception to processor", exc_info=True)

                # Pause the main thread BEFORE exiting.  If the fault rank
                # exits immediately (os._exit), its DP partner (e.g. rank 5
                # in DP group [1,5]) will hang on any in-progress collective
                # (error19).  By entering _handle_pause_and_dump the fault
                # rank completes its current ops and waits for the
                # Controller's EXIT signal after the dump finishes.
                _handle_pause_and_dump(processor)
                return None
            else:
                # Non-TTP error, re-raise
                raise
        except Exception as e:
            logger.error("[EXCEPTION_HANDLER] Caught Exception: %s: %s", type(e).__name__, e)

            # Check if this is a PAUSE-injected exception
            # PyThreadState_SetAsyncExc may inject non-RuntimeError types
            if processor is not None and processor._pause_type == 'RAISE':
                _handle_pause_and_dump(processor)

            # For other exceptions, attempt to report
            if processor is not None:
                try:
                    processor.on_worker_exception(e)
                    _send_stopped_heartbeat_and_exit(processor, exit_code=1)
                except Exception:
                    logger.warning("[TTP] Failed to handle worker exception", exc_info=True)

            raise

    return wrapper
