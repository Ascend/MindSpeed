# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import ctypes
import json
import logging
import os
import sys
import threading
import time
from typing import Callable, Optional, List, Dict


from ..config import TTPConfig
from ..constants import (
    MSG_TYPE_HEARTBEAT,
    MSG_TYPE_HEARTBEAT_ACK,
    MSG_TYPE_REGISTER,
    MSG_TYPE_REGISTER_ACK,
    MSG_TYPE_REGISTER_REPLICA_GROUP,
    MSG_TYPE_REGISTER_REPLICA_GROUP_ACK,
    MSG_TYPE_DUMP_REQUEST,
    MSG_TYPE_DUMP_RESPONSE,
    MSG_TYPE_WAIT_RELEASE,
    MSG_TYPE_EXIT,
    WorkerStatus,
)
from .heartbeat import HeartbeatManager
from .socket_client import SocketClient

logger = logging.getLogger(__name__)

os.environ['RAY_DEDUP_LOGS'] = '0'

RET_OK = 0
RET_ERROR = 1


class SaveHandler:
    """Manages checkpoint save callbacks and dump state"""

    def __init__(self):
        self._save_ckpt_callback: Optional[Callable] = None
        self._save_ckpt_ctx = None
        self._worker_instance = None
        self._need_dump = False
        self._dump_info = None
        self._started_dump = False
        self._lock = threading.Lock()
        self._optimizer_replica_info: List[dict] = []
        self._dump_world_ranks: Optional[List[int]] = None
        self._dump_cond = threading.Condition()
        self._dump_status = -1

    def set_worker_instance(self, worker) -> None:
        with self._lock:
            self._worker_instance = worker

    def get_worker_instance(self):
        with self._lock:
            return self._worker_instance

    def register_save_ckpt_handler(self, func: Callable, ctx=None) -> None:
        self._save_ckpt_callback = func
        self._save_ckpt_ctx = ctx

    def set_dump_world_ranks(self, world_ranks: List[int]) -> None:
        with self._lock:
            self._dump_world_ranks = world_ranks

    def get_dump_world_ranks(self) -> Optional[List[int]]:
        with self._lock:
            return self._dump_world_ranks

    def set_optimizer_replica(self, replica_info: List[dict]) -> None:
        with self._lock:
            self._optimizer_replica_info = replica_info

    def get_optimizer_replica(self) -> List[dict]:
        with self._lock:
            return self._optimizer_replica_info.copy()

    def set_dump_config(self, dump_info: list) -> int:
        with self._lock:
            if self._need_dump:
                logger.warning("already set dump config, but receive new dump config!")
                return RET_ERROR

            self._dump_info = dump_info
            self._need_dump = True
            self._started_dump = False
            return RET_OK

    def has_pending_dump(self) -> bool:
        with self._lock:
            return self._need_dump

    def reset_dump_state(self) -> None:
        with self._lock:
            self._need_dump = False
            self._started_dump = False
            self._dump_info = None

    def execute_save(self) -> bool:
        """Execute the save operation. Returns True on success, False otherwise."""
        with self._lock:
            if not self._need_dump:
                logger.info("[EXECUTE_SAVE_2] don't have dump config, nothing todo.")
                return False
            if self._started_dump:
                with self._dump_cond:
                    if self._dump_status == 0:
                        logger.info("[EXECUTE_SAVE_2] dump already completed successfully, returning True.")
                        return True
                    elif self._dump_status == 1:
                        logger.info("[EXECUTE_SAVE_2] dump already failed, returning False.")
                        return False
                logger.info("[EXECUTE_SAVE_2] dump in progress, returning False.")
                return False

            self._started_dump = True

        try:
            if self._save_ckpt_callback is not None:
                processor = TTPProcessor.get_instance()
                step = processor.current_iteration if processor else 0
                worker = self._worker_instance or self._save_ckpt_ctx

                success = self._save_ckpt_callback(step, self._dump_info, worker)

                return success
            else:
                logger.error("[EXECUTE_SAVE_ERROR] save_ckpt callback is not registered!")
                return False
        except Exception as e:
            logger.error("[EXECUTE_SAVE_ERROR] Failed to execute save: %s", e, exc_info=True)
            return False


class TTPProcessor:
    """TTP Processor (worker side)"""

    _instance: Optional['TTPProcessor'] = None
    _instance_lock = threading.Lock()

    STOP_TIMEOUT = 5.0

    def __init__(self, rank: int, world_size: int, config: TTPConfig):
        self.rank = rank
        self.world_size = world_size
        self.config = config

        self.client: Optional[SocketClient] = None
        self.heartbeat_manager: Optional[HeartbeatManager] = None
        self.replica_manager = None

        self._current_iteration: int = 0
        self._current_status: WorkerStatus = WorkerStatus.INIT
        self._is_inference_phase: bool = False
        self._worker_instance = None

        self.dump_request_event = threading.Event()
        self._dump_completed = False
        self._dump_ranks: list = []
        self._current_request_id: Optional[str] = None
        self._dump_group = None

        self._dump_cond = threading.Condition()
        self._dump_status = -1
        self._dump_timeout = 30 * 60
        self._dump_request_in_progress = False

        self.save_handler = SaveHandler()

        self._running = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        self._limit_step: int = 0
        self._pause_type: Optional[str] = None
        self._main_thread_id: Optional[int] = None

    @classmethod
    def get_instance(cls) -> Optional['TTPProcessor']:
        return cls._instance

    @classmethod
    def set_instance(cls, instance: 'TTPProcessor') -> None:
        with cls._instance_lock:
            cls._instance = instance
        from .._registry import set_processor

        set_processor(instance)

    @property
    def running(self) -> bool:
        return self._running and not self._stop_event.is_set()

    @property
    def current_iteration(self) -> int:
        with self._lock:
            return self._current_iteration

    @property
    def current_status(self) -> WorkerStatus:
        with self._lock:
            return self._current_status

    @property
    def is_inference_phase(self) -> bool:
        with self._lock:
            return self._is_inference_phase

    def start(self, server_ip: str, server_port: int) -> bool:
        try:
            self.client = SocketClient(server_ip, server_port)
            self.client.connect(max_retry_times=10, retry_interval=1.0)

            self.client.register_handler(MSG_TYPE_HEARTBEAT_ACK, self._on_heartbeat_ack)
            self.client.register_handler(MSG_TYPE_REGISTER_ACK, self._on_register_ack)
            self.client.register_handler(MSG_TYPE_REGISTER_REPLICA_GROUP_ACK, self._on_register_replica_group_ack)
            self.client.register_handler(MSG_TYPE_DUMP_REQUEST, self._on_dump_request)
            self.client.register_handler(MSG_TYPE_WAIT_RELEASE, self._on_wait_release)
            self.client.register_handler(MSG_TYPE_EXIT, self._on_exit)

            self.client.start_receive_loop()
            self._register_to_server()

            from ..replica.replica_group import ReplicaGroupManager

            self.replica_manager = ReplicaGroupManager(self.rank, self.world_size)
            self.replica_manager.build_groups_from_megatron()
            self._register_replica_groups()

            self.heartbeat_manager = HeartbeatManager(
                config=self.config.heartbeat, send_func=self._send_heartbeat_safe, on_timeout=self._on_heartbeat_timeout
            )
            self.heartbeat_manager.start()

            with self._lock:
                self._current_status = WorkerStatus.NORMAL
                self._running = True

            from .._registry import set_processor

            set_processor(self)

            self._stop_event.clear()
            self.dump_request_event.set()

            logger.warning("[TTP] TTPProcessor started successfully for rank %s", self.rank)
            return True
        except Exception as e:
            logger.error("[TTP] Failed to start TTPProcessor: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to start TTPProcessor: {e}")

    def stop(self) -> None:
        logger.warning("[TTP] Stopping TTPProcessor for rank %s", self.rank)

        self._stop_event.set()
        self._running = False

        if self.heartbeat_manager:
            self.heartbeat_manager.stop()

        if self.client:
            self.client.disconnect()

        self.dump_request_event.set()

        logger.warning("[TTP] TTPProcessor stopped for rank %s", self.rank)

    _heartbeat_fail_count: int = 0
    _HEARTBEAT_FAIL_LOG_MAX = 3

    def _send_heartbeat_safe(self, msg_type: int, data: bytes) -> bool:
        if self._stop_event.is_set():
            return False
        if self.client and self.client.connected:
            result = self.client.send_message(msg_type, data)
            if not result:
                if self._heartbeat_fail_count < self._HEARTBEAT_FAIL_LOG_MAX:
                    logger.error("Failed to send heartbeat message: msg_type=%s, data_len=%d", msg_type, len(data))
                    self._heartbeat_fail_count += 1
                elif self._heartbeat_fail_count == self._HEARTBEAT_FAIL_LOG_MAX:
                    self._heartbeat_fail_count += 1
                    logger.error("Failed to send heartbeat (suppressed, exceeding max log count)")
            return result
        return False

    def update_status(self, status: WorkerStatus, iteration: int = None) -> None:
        """Update worker status and optionally the current iteration"""
        with self._lock:
            self._current_status = status
            if iteration is not None:
                self._current_iteration = iteration

            if self.heartbeat_manager:
                self.heartbeat_manager.update_status(status, iteration)

    def update_iteration(self, iteration: int) -> None:
        """Update current iteration and check pause state"""
        with self._lock:
            self._current_iteration = iteration

            if self.heartbeat_manager:
                self.heartbeat_manager.update_iteration(iteration)

        self.tft_pause_train(iteration)

    def tft_pause_train(self, cur_step: int) -> None:
        """Check pause state and raise STEP FINISH if at limit step"""
        if cur_step < 0:
            return

        with self._lock:
            limit_step = self._limit_step
            pause_type = self._pause_type

        if pause_type in ['PAUSE', 'RAISE'] and cur_step == limit_step:
            if pause_type == 'RAISE':
                raise RuntimeError("STEP FINISH")

    def check_and_raise_if_paused(self) -> None:
        """Raise STEP FINISH if PAUSE/RAISE is active"""
        with self._lock:
            pause_type = self._pause_type

        if pause_type in ['PAUSE', 'RAISE']:
            if pause_type == 'RAISE':
                raise RuntimeError("STEP FINISH")

    def set_main_thread_id(self, thread_id: Optional[int] = None) -> None:
        """Set the main thread ID for async exception injection"""
        if thread_id is None:
            thread_id = threading.current_thread().ident

        with self._lock:
            self._main_thread_id = thread_id

        # Install a global excepthook to catch FORCE STOP that escapes
        # @ttp_exception_handler.  stop_device in tft_save_callback kills
        # in-flight NPU ops on ranks that missed PAUSE (still in data.to()
        # or forward pass).  @ttp_exception_handler only wraps optimizer
        # methods and forward_backward_no_pipelining — it does NOT cover
        # the preamble (data.to()) at the start of forward_backward_batch.
        # This hook catches FORCE STOP (and similar TTP errors) anywhere
        # in the main thread, routes them to _handle_pause_and_dump, and
        # prevents the worker from crashing.
        _processor = self
        _orig_excepthook = sys.excepthook

        def _ttp_excepthook(exc_type, exc_value, exc_tb):
            _err_str = str(exc_value) if exc_value else ''
            # Check for TTP-handled errors FIRST (before _pause_type check),
            # otherwise FORCE STOP on a rank with _pause_type=='RAISE' would
            # be silently passed to the original hook.
            if exc_type is RuntimeError and any(
                s in _err_str
                for s in [
                    'FORCE STOP',
                    'UCE',
                    'HBM',
                    'HCCL',
                    'NCCL',
                    'TTP_INJECTED_FAULT',
                ]
            ):
                from mindspeed.core.ttp.recovery.exception_handler import _handle_pause_and_dump

                try:
                    _handle_pause_and_dump(_processor)
                except Exception:
                    logger.warning("[TTP] Failed to handle pause and dump in excepthook", exc_info=True)
                return
            _orig_excepthook(exc_type, exc_value, exc_tb)

        sys.excepthook = _ttp_excepthook

    def inject_exception_to_main_thread(self, exception: Exception = RuntimeError("STEP FINISH")) -> bool:
        """Inject an exception into the main thread via PyThreadState_SetAsyncExc.

        Used to interrupt training with STEP FINISH when the Controller sends a PAUSE.
        Returns True if the exception was successfully injected.
        """
        with self._lock:
            main_thread_id = self._main_thread_id

        if main_thread_id is None:
            return False

        try:
            result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(main_thread_id), ctypes.py_object(type(exception))
            )

            if result == 0:
                return False
            elif result > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(main_thread_id), None)
                return False
            else:
                return True

        except Exception:
            return False

    def set_inference_phase(self, is_inference: bool) -> None:
        """Toggle inference phase mode (no heartbeats during inference)"""
        with self._lock:
            self._is_inference_phase = is_inference

    def check_and_wait_if_paused(self, timeout: float = 300.0) -> bool:
        """Wait if the Controller has paused this worker. Returns True if proceeding is ok."""
        if self._stop_event.is_set():
            return False

        if not self.dump_request_event.is_set():
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.dump_request_event.wait(timeout=0.1):
                    return True

                if self._stop_event.is_set():
                    return False

            return False

        return True

    def get_dump_ranks(self) -> list:
        with self._lock:
            return self._dump_ranks.copy()

    def notify_dump_completed(self, success: bool, iteration: int) -> None:
        """Notify the Controller that the dump operation has completed"""
        with self._lock:
            self._dump_completed = success

        if self.client and self.client.connected:
            response_data = json.dumps(
                {
                    'rank': self.rank,
                    'success': success,
                    'iteration': iteration,
                    'timestamp': time.time(),
                }
            ).encode('utf-8')

            self.client.send_message(MSG_TYPE_DUMP_RESPONSE, response_data)
            logger.warning(
                "[TTP] [NOTIFY_DUMP_3] Notified dump completed: success=%s, iteration=%s", success, iteration
            )
        else:
            pass

    def _register_to_server(self) -> bool:
        register_data = json.dumps(
            {
                'rank': self.rank,
                'world_size': self.world_size,
                'timestamp': time.time(),
            }
        ).encode('utf-8')

        logger.debug("Registering to server at %s:%s", self.config.server_ip, self.config.server_port)
        success, _ = self.client.send_message_with_response(MSG_TYPE_REGISTER, register_data, timeout=30.0)
        if success:
            logger.warning("[TTP] Successfully registered to server")
        else:
            logger.warning("Failed to register to server")
        return success

    def _register_replica_groups(self) -> bool:
        if not self.replica_manager:
            return False

        replica_data = self.replica_manager.to_dict()

        replica_data_json = json.dumps(replica_data).encode('utf-8')

        logger.debug("Registering replica groups to server")
        self.client.send_message_with_response(MSG_TYPE_REGISTER_REPLICA_GROUP, replica_data_json, timeout=30.0)
        return True

    def _on_heartbeat_ack(self, data: bytes) -> None:
        pass

    def _on_register_ack(self, data: bytes) -> None:
        pass

    def _on_register_replica_group_ack(self, data: bytes) -> None:
        pass

    def _on_dump_request(self, data: bytes) -> None:
        with self._lock:
            if self._dump_request_in_progress:
                return
            self._dump_request_in_progress = True

        try:
            ckpt_msg = json.loads(data.decode('utf-8'))

            step = ckpt_msg.get('step', 0)
            if step <= 0:
                self._send_dump_reply(ckpt_msg, success=False)
                return

            rank_vec = []
            group_idx = []
            found_count = 0
            p_idx = 0

            num = ckpt_msg.get('num', 0)

            for i in range(num):
                group_idx.append(ckpt_msg['ranks'][p_idx])
                p_idx += 1
                rank_list_size = ckpt_msg['ranks'][p_idx]
                p_idx += 1
                tmp_rank_list = []
                for j in range(rank_list_size):
                    tmp_rank_list.append(ckpt_msg['ranks'][p_idx])
                    p_idx += 1
                rank_vec.append(tmp_rank_list)

                if self.rank in tmp_rank_list:
                    found_count += 1

            world_ranks = ckpt_msg.get('world_ranks', [])

            if found_count != num:
                self._send_dump_reply(ckpt_msg, success=False)
                return

            with self._lock:
                self._current_iteration = step
                self._current_request_id = ckpt_msg.get('repair_id', '')

            try:
                save_info = []
                for list_idx, rank_list in enumerate(rank_vec):
                    save_info.append({"type": group_idx[list_idx], "ranks": rank_list, "world_ranks": world_ranks})

                self.save_handler.set_dump_world_ranks(world_ranks)

                self.save_handler.set_dump_config(save_info)

                with self._dump_cond:
                    self._dump_status = -1

                with self.save_handler._dump_cond:
                    self.save_handler._dump_status = -1
                    self.save_handler._started_dump = False

                # Signal wait_next_action so the main thread (if already in
                # _handle_pause_and_dump) returns "DUMP" and enters
                # wait_for_dump_complete.  Spawn the save thread immediately
                # — stop_device inside tft_save_callback will break any
                # stuck NCCL collectives on the main thread.  This matches
                # the proven stable reference implementation.
                self.dump_request_event.set()

                save_thread = threading.Thread(
                    target=self._save_data_thread, args=(ckpt_msg,), daemon=True, name=f"save_thread_rank{self.rank}"
                )
                save_thread.start()

                logger.warning("[TTP] [DUMP_REQUEST] Received dump request: step=%s, world_ranks=%s", step, world_ranks)

            finally:
                pass

        except Exception:
            try:
                ckpt_msg = json.loads(data.decode('utf-8'))
                self._send_dump_reply(ckpt_msg, success=False)
            except Exception:
                logger.warning("[TTP] Failed to send dump reply in fallback", exc_info=True)
            with self._lock:
                self._dump_request_in_progress = False

    def _on_wait_release(self, data: bytes) -> None:
        try:
            request = json.loads(data.decode('utf-8'))
            action = request.get('action', '')
            request_id = request.get('request_id', '')

            if action == 'PAUSE':
                # Immediately report PAUSE to Controller — matching the stable
                # reference implementation (de8fa3fd).  This lets the Controller
                # know the rank received the PAUSE command and is preparing to
                # stop.  The main thread will be interrupted asynchronously via
                # inject_exception_to_main_thread().  If the main thread is stuck
                # in NCCL, stop_device in tft_save_callback will break it free.
                self.update_status(WorkerStatus.PAUSE)

                if request_id:
                    with self._lock:
                        self._current_request_id = request_id

                self.dump_request_event.clear()

                with self._lock:
                    self._limit_step = self._current_iteration + 1
                    self._pause_type = 'RAISE'

                self.inject_exception_to_main_thread()
            else:
                self.dump_request_event.set()

        except Exception as e:
            logger.error("[WAIT_RELEASE_ERROR] Failed to handle WAIT_RELEASE: %s", e, exc_info=True)

    def _send_heartbeat_now(self) -> None:
        try:
            with self._lock:
                status = self._current_status
                iteration = self._current_iteration
            heartbeat_data = json.dumps(
                {
                    'status': int(status),
                    'iteration': iteration,
                    'timestamp': time.time(),
                }
            ).encode('utf-8')
            self._send_heartbeat_safe(MSG_TYPE_HEARTBEAT, heartbeat_data)
        except Exception:
            logger.warning("[TTP] Failed to send heartbeat", exc_info=True)

    def _on_exit(self, data: bytes) -> None:
        try:
            try:
                exit_info = json.loads(data.decode('utf-8'))  # noqa: F841
            except Exception:
                logger.warning("[TTP] Failed to parse exit info", exc_info=True)

            self._stop_event.set()
            self.dump_request_event.set()

            try:
                if self.heartbeat_manager and self.heartbeat_manager.running:
                    self.heartbeat_manager.update_status(WorkerStatus.STOPPED)
                    self._send_heartbeat_now()
            except Exception:
                logger.warning("[TTP] Failed to send stopped heartbeat on exit", exc_info=True)

            def _exit_thread():
                time.sleep(3)
                os._exit(0)

            t = threading.Thread(target=_exit_thread, daemon=True)
            t.start()

        except Exception as e:
            logger.error("[EXIT_ERROR] Failed to handle EXIT: %s", e, exc_info=True)
            os._exit(1)

    def _on_heartbeat_timeout(self, missed_count: int) -> None:
        logger.warning("Heartbeat timeout, missed count: %s", missed_count)

    def _save_data_thread(self, ckpt_msg: Dict):
        """Background thread that executes the save and sends the dump reply to the Controller

        Runs ONLY on dump ranks (selected by the Controller in begin_exception_ckpt).

        Matches the mindio-ttp reference implementation: the daemon delegates
        directly to the save callback (tft_save_callback / execute_save).
        NPU stop-clean is handled inside tft_save_callback, where it runs
        after the main thread has paused — so stop_device is a no-op when
        everything works correctly.
        """
        try:
            success = False

            success = self.save_handler.execute_save()

        except Exception:
            success = False
        finally:
            self._dump_group = None

            if not success:
                self.save_handler.reset_dump_state()

            if ckpt_msg:
                self._send_dump_reply(ckpt_msg, success=success)

            with self._dump_cond:
                self._dump_status = 0 if success else 1
                self._dump_cond.notify_all()

            with self.save_handler._dump_cond:
                self.save_handler._dump_status = 0 if success else 1
                self.save_handler._dump_cond.notify_all()

            if self.client and self.client.connected:
                if self.client.receive_thread is None or not self.client.receive_thread.is_alive():
                    self.client.start_receive_loop()

    def _send_dump_reply(self, ckpt_msg: Dict, success: bool) -> None:
        logger.error(
            "[TTP] [SEND_DUMP_REPLY] rank:%s sending dump reply: success=%s, step=%s",
            self.rank,
            success,
            ckpt_msg.get('step', 0),
        )

        reply_msg = {
            'rank': self.rank,
            'step': ckpt_msg.get('step', 0),
            'iteration': ckpt_msg.get('step', 0),
            'sn': ckpt_msg.get('sn', 0),
            'repair_id': ckpt_msg.get('repair_id', ''),
            'success': success,
            'timestamp': time.time(),
        }

        reply_data = json.dumps(reply_msg).encode('utf-8')
        if self.client and self.client.connected:
            self.client.send_message(MSG_TYPE_DUMP_RESPONSE, reply_data)
            logger.warning("[TTP] [DUMP_REPLY] Sent dump reply: success=%s, step=%s", success, reply_msg['step'])
        else:
            logger.error("[DUMP_REPLY_ERROR] Client not connected, cannot send dump reply")

    def report_error(self, error_type: str = "UNKNOWN") -> bool:
        """Report an error to the Controller by marking ABNORMAL and sending a heartbeat"""
        self.update_status(WorkerStatus.ABNORMAL)

        if self.heartbeat_manager:
            try:
                self.heartbeat_manager._send_heartbeat()
                return True
            except Exception:
                return False
        else:
            return False

    def wait_next_action(self, timeout: float = 300.0) -> str:
        """Wait for the next action from the Controller. Returns 'DUMP', 'EXIT', or 'TIMEOUT'."""
        self._enter_wait_mode()

        try:
            start_time = time.time()
            check_count = 0
            while time.time() - start_time < timeout:
                check_count += 1

                if self._stop_event.is_set():
                    return "EXIT"

                if self.dump_request_event.is_set():
                    return "DUMP"

                time.sleep(0.1)

            return "TIMEOUT"
        finally:
            self._exit_wait_mode()

    def _enter_wait_mode(self) -> None:
        self.update_status(WorkerStatus.PAUSE)

        if self.client and self.client.connected:
            if self.client.receive_thread is None or not self.client.receive_thread.is_alive():
                self.client.start_receive_loop()

    def _exit_wait_mode(self) -> None:
        self.dump_request_event.clear()

    def wait_for_dump_complete(self, timeout: float = 1800.0) -> bool:
        """Wait for the dump thread to finish. Returns True on success, False on failure."""
        start_time = time.time()
        check_count = 0
        while time.time() - start_time < timeout:
            check_count += 1

            with self._dump_cond:
                if self._dump_status == 0:
                    return True
                elif self._dump_status == 1:
                    return False

            if self._stop_event.is_set():
                with self._dump_cond:
                    if self._dump_status == -1:
                        return True
                return False

            if check_count % 100 == 0:
                _elapsed = time.time() - start_time

            time.sleep(0.1)

        return False

    def on_worker_exception(self, exception: Exception) -> str:
        """Handle a worker exception: report to Controller, then wait for dump or exit."""
        error_str = str(exception)

        error_type = "UNKNOWN"
        if "UCE" in error_str or "HBM" in error_str:
            error_type = "UCE"
        elif "HCCL" in error_str or "NCCL" in error_str:
            error_type = "HCCL_FAILED"
        elif "RuntimeError" in error_str:
            error_type = "RUNTIME_ERROR"

        self.report_error(error_type)

        try:
            start_time = time.time()
            timeout = 600.0

            while time.time() - start_time < timeout:
                if self._stop_event.is_set():
                    os._exit(0)

                if self.save_handler.has_pending_dump():
                    self.save_handler.execute_save()
                    self.save_handler.reset_dump_state()

                time.sleep(0.1)

            os._exit(1)

        except Exception:
            os._exit(1)

        return "EXIT"


def register_save_ckpt_handler(func: Callable, ctx=None) -> None:
    processor = TTPProcessor.get_instance()
    if processor:
        processor.save_handler.register_save_ckpt_handler(func, ctx)


def set_worker_instance(worker) -> None:
    processor = TTPProcessor.get_instance()
    if processor:
        processor.save_handler.set_worker_instance(worker)


def get_processor() -> Optional[TTPProcessor]:
    return TTPProcessor.get_instance()


def get_worker_instance():
    processor = TTPProcessor.get_instance()
    if processor:
        return processor.save_handler.get_worker_instance()
    return None
