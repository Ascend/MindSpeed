# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from typing import Dict, List, Optional, Set

from ..config import TTPConfig, get_ttp_config
from ..constants import (
    MSG_TYPE_HEARTBEAT,
    MSG_TYPE_HEARTBEAT_ACK,
    MSG_TYPE_REGISTER,
    MSG_TYPE_REGISTER_ACK,
    MSG_TYPE_REGISTER_REPLICA_GROUP,
    MSG_TYPE_REGISTER_REPLICA_GROUP_ACK,
    MSG_TYPE_DUMP_REQUEST,
    MSG_TYPE_DUMP_RESPONSE,
    MSG_TYPE_EXCEPTION,
    MSG_TYPE_TRIGGER_DUMP,
    MSG_TYPE_TRIGGER_DUMP_ACK,
    MSG_TYPE_WAIT_RELEASE,
    MSG_TYPE_EXIT,
    WorkerStatus,
)
from .heartbeat import HeartbeatChecker
from .socket_server import ClientConnection, SocketServer

logger = logging.getLogger(__name__)


class TTPController:
    """TTP Controller (server side)"""

    _instance: Optional['TTPController'] = None
    _instance_lock = threading.Lock()

    STOP_TIMEOUT = 5.0

    def __init__(self, rank: int, world_size: int, config: TTPConfig):
        self.rank = rank
        self.world_size = world_size
        self.config = config

        self.server: Optional[SocketServer] = None
        self.heartbeat_checker: Optional[HeartbeatChecker] = None
        self.replica_manager = None

        self._worker_connections: Dict[int, ClientConnection] = {}
        self._worker_status: Dict[int, WorkerStatus] = {}
        self._worker_iteration: Dict[int, int] = {}

        self._fault_ranks: Set[int] = set()
        self._dump_in_progress: bool = False
        self._dump_lock = threading.Lock()
        self._dump_complete_event = threading.Event()
        self._dump_success = False
        self._dump_completed = False
        self._pending_wait_release_ranks: List[int] = []
        self._current_dump_ranks: List[int] = []
        self._dump_completed_ranks: set = set()
        self._dump_success_map: dict = {}
        self._current_request_id: Optional[str] = None
        self._request_timestamp: float = 0

        self._ready = threading.Event()
        self._dump_iteration: int = 0

        self._running = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

    @classmethod
    def get_instance(cls) -> Optional['TTPController']:
        """Get the singleton instance"""
        return cls._instance

    @classmethod
    def set_instance(cls, instance: 'TTPController') -> None:
        """Set the singleton instance"""
        with cls._instance_lock:
            cls._instance = instance

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Wait until the Controller server is ready to accept connections."""
        return self._ready.wait(timeout=timeout)

    @property
    def running(self) -> bool:
        """Whether the controller is running"""
        return self._running and not self._stop_event.is_set()

    @property
    def worker_status(self) -> Dict[int, WorkerStatus]:
        """Worker status dictionary (read-only)"""
        with self._lock:
            return self._worker_status.copy()

    def start(self, host: str, port: int) -> bool:
        """Start the controller"""
        try:
            logger.warning("[TTP] Starting TTPController on %s:%s", host, port)

            self.server = SocketServer(host, port)

            self.server.register_handler(MSG_TYPE_REGISTER, self._on_register)
            self.server.register_handler(MSG_TYPE_REGISTER_REPLICA_GROUP, self._on_register_replica_group)
            self.server.register_handler(MSG_TYPE_HEARTBEAT, self._on_heartbeat)
            self.server.register_handler(MSG_TYPE_EXCEPTION, self._on_exception)
            self.server.register_handler(MSG_TYPE_DUMP_RESPONSE, self._on_dump_response)
            self.server.register_handler(MSG_TYPE_TRIGGER_DUMP, self._on_trigger_dump)

            self.server.register_disconnect_handler(self._on_disconnect)

            self.server.start()
            self._ready.set()

            from ..replica.replica_group import ServerReplicaGroupManager

            self.replica_manager = ServerReplicaGroupManager()

            self.heartbeat_checker = HeartbeatChecker(
                config=self.config.heartbeat, on_timeout=self._on_worker_timeout, world_size=self.world_size
            )
            self.heartbeat_checker.start()

            self._running = True
            self._stop_event.clear()

            logger.warning("[TTP] TTPController started successfully on %s:%s", host, port)
            return True
        except Exception as e:
            logger.error("[TTP] Failed to start TTPController: %s", e, exc_info=True)
            raise RuntimeError(f"[TTP] Failed to start TTPController: {e}")

    def stop(self) -> None:
        """Stop the controller (graceful shutdown)"""
        logger.warning("[TTP] Stopping TTPController")

        self._stop_event.set()
        self._running = False

        if self.heartbeat_checker:
            logger.debug("Stopping heartbeat checker")
            self.heartbeat_checker.stop()

        if self.server:
            logger.debug("Stopping server")
            self.server.stop()

        logger.warning("[TTP] TTPController stopped")

    def _on_register(self, conn: ClientConnection, data: bytes) -> None:
        """Handle registration request"""
        try:
            request = json.loads(data.decode('utf-8'))
            rank = request['rank']
            world_size = request['world_size']  # noqa: F841

            conn.rank = rank

            with self._lock:
                self._worker_status[rank] = WorkerStatus.NORMAL
                self._worker_connections[rank] = conn

            if self.heartbeat_checker:
                self.heartbeat_checker.register_worker(rank)

            sn = request.get('sn')

            ack_data_dict = {
                'success': True,
                'rank': rank,
                'timestamp': time.time(),
            }
            if sn is not None:
                ack_data_dict['sn'] = sn

            ack_data = json.dumps(ack_data_dict).encode('utf-8')
            conn.send_message(MSG_TYPE_REGISTER_ACK, ack_data)

        except Exception as e:
            logger.error("Failed to handle register: %s", e, exc_info=True)

            try:
                request = json.loads(data.decode('utf-8'))
                sn = request.get('sn')
            except Exception:
                sn = None

            ack_data_dict = {
                'success': False,
                'error': str(e),
            }
            if sn is not None:
                ack_data_dict['sn'] = sn

            ack_data = json.dumps(ack_data_dict).encode('utf-8')
            conn.send_message(MSG_TYPE_REGISTER_ACK, ack_data)

    def _on_register_replica_group(self, conn: ClientConnection, data: bytes) -> None:
        """Handle replica group registration"""
        try:
            request = json.loads(data.decode('utf-8'))
            rank = request['rank']

            if self.replica_manager:
                self.replica_manager.update_from_worker(rank, request)

            logger.debug("Replica group registered for rank %s", rank)

            sn = request.get('sn')
            ack_data = json.dumps(
                {
                    'success': True,
                    'rank': rank,
                    'sn': sn,
                }
            ).encode('utf-8')
            conn.send_message(MSG_TYPE_REGISTER_REPLICA_GROUP_ACK, ack_data)

        except Exception as e:
            logger.error("Failed to handle replica group register: %s", e, exc_info=True)

            sn = request.get('sn') if 'request' in locals() else None
            ack_data = json.dumps(
                {
                    'success': False,
                    'error': str(e),
                    'sn': sn,
                }
            ).encode('utf-8')
            conn.send_message(MSG_TYPE_REGISTER_REPLICA_GROUP_ACK, ack_data)

    def _on_heartbeat(self, conn: ClientConnection, data: bytes) -> None:
        """Handle heartbeat"""
        try:
            request = json.loads(data.decode('utf-8'))
            rank = conn.rank

            status_value = request.get('status', WorkerStatus.NORMAL)
            iteration = request.get('iteration', 0)

            if isinstance(status_value, int):
                status = WorkerStatus(status_value)
            elif status_value == 'ABNORMAL':
                status = WorkerStatus.ABNORMAL
            elif status_value == 'PAUSE':
                status = WorkerStatus.PAUSE
            elif status_value == 'FAULT':
                status = WorkerStatus.FAULT
            elif status_value == 'STOPPED':
                status = WorkerStatus.STOPPED
            else:
                status = WorkerStatus.NORMAL

            with self._lock:
                old_status = self._worker_status.get(rank, None)
                if old_status in (WorkerStatus.STOPPED, WorkerStatus.FAULT):
                    if self.heartbeat_checker:
                        self.heartbeat_checker.update_heartbeat(rank, old_status)
                else:
                    self._worker_status[rank] = status
                if iteration > 0:
                    self._worker_iteration[rank] = iteration

            if self.heartbeat_checker:
                self.heartbeat_checker.update_heartbeat(rank, status)

            if self.replica_manager:
                self.replica_manager.update_worker_status(rank, status)

            sn = request.get('sn')

            ack_data_dict = {
                'success': True,
                'timestamp': time.time(),
            }
            if sn is not None:
                ack_data_dict['sn'] = sn

            ack_data = json.dumps(ack_data_dict).encode('utf-8')
            conn.send_message(MSG_TYPE_HEARTBEAT_ACK, ack_data)

            if status == WorkerStatus.ABNORMAL:
                with self._lock:
                    current_status = self._worker_status.get(rank, None)
                if current_status not in (WorkerStatus.STOPPED, WorkerStatus.FAULT):
                    self._on_worker_fault(rank)

        except Exception as e:
            logger.error("Failed to handle heartbeat: %s", e, exc_info=True)

    def _on_exception(self, conn: ClientConnection, data: bytes) -> None:
        """Handle exception report"""
        try:
            request = json.loads(data.decode('utf-8'))
            rank = request['rank']
            exception_type = request['exception_type']
            exception_msg = request['exception_msg']
            iteration = request['iteration']

            with self._lock:
                self._worker_status[rank] = WorkerStatus.ABNORMAL
                self._worker_iteration[rank] = iteration

            if self.replica_manager:
                self.replica_manager.update_worker_status(rank, WorkerStatus.ABNORMAL)

            logger.error(
                "Exception from rank %s: %s: %s (iteration=%s)", rank, exception_type, exception_msg, iteration
            )
            self._on_worker_fault(rank)

        except Exception as e:
            logger.error("Failed to handle exception report: %s", e, exc_info=True)

    def _on_dump_response(self, conn: ClientConnection, data: bytes) -> None:
        """Handle dump response"""
        try:
            request = json.loads(data.decode('utf-8'))
            rank = request['rank']
            success = request['success']
            iteration = request['iteration']  # noqa: F841

            dump_ranks = self._current_dump_ranks

            if rank not in dump_ranks:
                return

            self._dump_completed_ranks.add(rank)
            self._dump_success_map[rank] = success

            all_dump_completed = len(self._dump_completed_ranks) == len(dump_ranks)

            if all_dump_completed:
                all_dump_success = all(self._dump_success_map.get(r, False) for r in dump_ranks)

                dump_check_success = True
                if all_dump_success:
                    try:
                        dump_dir = self.config.dump_dir
                        dump_iteration = 0
                        with self._lock:
                            for r in dump_ranks:
                                if r in self._worker_iteration:
                                    dump_iteration = self._worker_iteration[r]
                                    break
                        self._dump_iteration = dump_iteration

                        if dump_iteration > 0 and os.path.exists(dump_dir):
                            dump_path = os.path.join(dump_dir, f"global_step_{dump_iteration}")
                            if not os.path.exists(dump_path):
                                dump_check_success = False
                    except Exception:
                        dump_check_success = False

                final_success = all_dump_success and dump_check_success
                with self._dump_lock:
                    self._dump_in_progress = False
                    self._dump_completed = True
                    self._dump_success = final_success

                pending_wait_release_ranks = getattr(self, '_pending_wait_release_ranks', [])
                if pending_wait_release_ranks:
                    for wait_rank in pending_wait_release_ranks:
                        release_data = json.dumps(
                            {
                                'action': 'WAIT_RELEASE',
                                'timestamp': time.time(),
                            }
                        ).encode('utf-8')

                        logger.info("Sending WAIT_RELEASE to rank %s after dump complete", wait_rank)
                        self.server.send_to_client(wait_rank, MSG_TYPE_WAIT_RELEASE, release_data)

                    self._pending_wait_release_ranks = []
                    self._dump_completed_ranks = set()
                    self._dump_success_map = {}

                if final_success:
                    connected_ranks = self.server.get_connected_ranks()

                    exit_sent_count = 0
                    for exit_rank in connected_ranks:
                        exit_data = json.dumps(
                            {
                                'action': 'EXIT',
                                'reason': 'dump_completed',
                                'timestamp': time.time(),
                            }
                        ).encode('utf-8')

                        success = self.server.send_to_client(exit_rank, MSG_TYPE_EXIT, exit_data)
                        if success:
                            exit_sent_count += 1

                    self._wait_for_stopped_confirmation(connected_ranks, timeout=10.0)
                else:
                    logger.error("[TTP] [DUMP_FAILED] Dump failed, sending EXIT to all ranks")

                    connected_ranks = self.server.get_connected_ranks()

                    exit_sent_count = 0
                    for exit_rank in connected_ranks:
                        exit_data = json.dumps(
                            {
                                'action': 'EXIT',
                                'reason': 'dump_failed',
                                'timestamp': time.time(),
                            }
                        ).encode('utf-8')

                        success = self.server.send_to_client(exit_rank, MSG_TYPE_EXIT, exit_data)
                        if success:
                            exit_sent_count += 1

                    self._wait_for_stopped_confirmation(connected_ranks, timeout=10.0)

                self._dump_complete_event.set()

        except Exception as e:
            logger.error("[DUMP_RESPONSE_ERROR] Failed to handle dump response: %s", e, exc_info=True)
            self._dump_complete_event.set()

    def _wait_for_stopped_confirmation(self, ranks: set, timeout: float = 10.0) -> None:
        """After sending EXIT, wait for all ranks to report STOPPED via heartbeat"""

        start_time = time.time()
        check_interval = 0.5
        pending = set(ranks)
        retry_count = 0
        max_retries = 2

        while pending and (time.time() - start_time) < timeout:
            with self._lock:
                confirmed = set()
                for rank in pending:
                    status = self._worker_status.get(rank)
                    if status in (WorkerStatus.STOPPED, WorkerStatus.FAULT):
                        confirmed.add(rank)
            pending -= confirmed
            if not pending:
                break

            elapsed = time.time() - start_time
            if retry_count < max_retries and elapsed > (timeout * (retry_count + 1) / (max_retries + 1)):
                retry_count += 1
                for retry_rank in pending:
                    exit_data = json.dumps(
                        {
                            'action': 'EXIT',
                            'reason': 'retry',
                            'timestamp': time.time(),
                        }
                    ).encode('utf-8')
                    self.server.send_to_client(retry_rank, MSG_TYPE_EXIT, exit_data)

            time.sleep(check_interval)

        if pending:
            len_pending = len(pending)
            logger.warning("[TTP] [EXIT_WAIT_TIMEOUT] %d ranks did not confirm STOPPED: %s", len_pending, pending)
            with self._lock:
                for rank in pending:
                    self._worker_status[rank] = WorkerStatus.FAULT

    def _on_disconnect(self, rank: int) -> None:
        """Handle disconnection"""
        with self._lock:
            current_status = self._worker_status.get(rank)

            if current_status == WorkerStatus.FAULT:
                logger.debug("[DISCONNECT] Worker rank %s already marked as FAULT, skipping", rank)
                if self.heartbeat_checker:
                    self.heartbeat_checker.unregister_worker(rank)
                return
            self._worker_status[rank] = WorkerStatus.FAULT

        if self.replica_manager:
            self.replica_manager.update_worker_status(rank, WorkerStatus.FAULT)

        self._on_worker_fault(rank)

        if self.heartbeat_checker:
            self.heartbeat_checker.unregister_worker(rank)

    def _on_worker_timeout(self, rank: int) -> None:
        """Handle worker timeout"""
        with self._lock:
            current_status = self._worker_status.get(rank, WorkerStatus.NORMAL)

        if current_status in (WorkerStatus.PAUSE, WorkerStatus.ABNORMAL):
            return

        with self._lock:
            self._worker_status[rank] = WorkerStatus.FAULT

        self._on_worker_fault(rank)

    def _on_worker_fault(self, fault_rank: int) -> None:
        """Handle worker fault with pause→dump→exit/restart flow.

        1. Pause all other workers via WAIT_RELEASE message.
        2. Build a dump checkpoint on healthy replica ranks.
        3. Send EXIT to the fault rank; on dump failure, exit the entire program.
        """

        with self._dump_lock:
            # Status check under dump_lock to avoid TOCTOU between _lock and _dump_lock
            current_status = self._worker_status.get(fault_rank)
            if current_status in (WorkerStatus.FAULT, WorkerStatus.STOPPED):
                return

            if self._dump_completed:
                return

            if self._dump_in_progress:
                if fault_rank not in self._fault_ranks:
                    self._fault_ranks.add(fault_rank)
                return

            if fault_rank in self._fault_ranks:
                return

            self._current_request_id = str(uuid.uuid4())
            self._request_timestamp = time.time()

            self._fault_ranks.add(fault_rank)
            self._dump_in_progress = True

        all_ranks = list(self._worker_status.keys())

        pause_sent_count = 0
        for rank in all_ranks:
            if rank != fault_rank:
                pause_data = json.dumps(
                    {
                        'action': 'PAUSE',
                        'fault_rank': fault_rank,
                        'request_id': self._current_request_id,
                        'timestamp': time.time(),
                    }
                ).encode('utf-8')

                success = self.server.send_to_client(rank, MSG_TYPE_WAIT_RELEASE, pause_data)
                if success:
                    pause_sent_count += 1

        wait_start = time.time()
        max_wait_time = 10.0
        all_paused = False

        while time.time() - wait_start < max_wait_time:
            with self._lock:
                paused_count = sum(1 for s in self._worker_status.values() if s == WorkerStatus.PAUSE)

            if paused_count >= len(all_ranks) - 1:
                all_paused = True
                break

            time.sleep(0.1)

        if not all_paused:
            with self._lock:
                status_str = ", ".join(["rank%s:%s" % (r, s.name) for r, s in self._worker_status.items()])
            logger.error("[TTP] [FAULT_HANDLER_PAUSE_TIMEOUT] Not all workers entered PAUSE: %s", status_str)

        ret = self.begin_exception_ckpt(self._fault_ranks)
        if ret != 0:
            with self._dump_lock:
                self._dump_in_progress = False
                self._fault_ranks.clear()

            logger.error(
                "[TTP] [FAULT_HANDLER_DUMP_SELECT_FAILED] No healthy replica group found, sending EXIT to all workers"
            )
            connected_ranks = self.server.get_connected_ranks()
            for exit_rank in connected_ranks:
                exit_data = json.dumps(
                    {
                        'action': 'EXIT',
                        'reason': 'dump_select_failed',
                        'timestamp': time.time(),
                    }
                ).encode('utf-8')
                self.server.send_to_client(exit_rank, MSG_TYPE_EXIT, exit_data)
        else:
            event_set = self._dump_complete_event.wait(timeout=600.0)

            fault_ranks_to_exit = list(self._fault_ranks)

            if event_set:
                with self._dump_lock:
                    actual_dump_success = self._dump_success

                if actual_dump_success:
                    dump_success = True
                else:
                    logger.error("[TTP] [FAULT_HANDLER_DUMP_FAILED] Dump completed but failed")
                    dump_success = False

                    logger.error("[TTP] [FAULT_HANDLER_EXIT_ON_FAILURE] Dump failed, exiting program")
                    self.stop()
                    sys.exit(1)
            else:
                logger.error("[TTP] [FAULT_HANDLER_DUMP_TIMEOUT] Dump timed out after 600s")
                dump_success = False

                logger.error("[TTP] [FAULT_HANDLER_EXIT_ON_TIMEOUT] Dump timeout, exiting program")
                self.stop()
                sys.exit(1)

            for exit_rank in fault_ranks_to_exit:
                exit_data = json.dumps(
                    {
                        'action': 'EXIT',
                        'reason': 'fault_handled',
                        'request_id': self._current_request_id,
                        'timestamp': time.time(),
                    }
                ).encode('utf-8')

                success = self.server.send_to_client(exit_rank, MSG_TYPE_EXIT, exit_data)
                if success:
                    with self._lock:
                        self._worker_status[fault_rank] = WorkerStatus.STOPPED

            with self._dump_lock:
                self._dump_in_progress = False
                self._fault_ranks.clear()
                self._dump_complete_event.clear()

            if not dump_success:
                with self._lock:
                    for exit_rank in fault_ranks_to_exit:
                        self._worker_status[exit_rank] = WorkerStatus.FAULT
                logger.error(
                    "[TTP] [FAULT_HANDLER_FAILED] Dump failed after timeout, marked fault ranks as FAULT status"
                )

    def choose_rank_inner_rl(self, rank_choose_info: Dict, tmp_rank_vec: List[int], rep_cnt: int) -> int:
        """Select a complete, healthy replica group for dump.

        'RL' refers to RL training topology where the world is split into two halves
        (e.g. actor/critic). Fault ranks in one half require selecting the other half.
        Returns TTP_OK (0) if a healthy group is found, TTP_ERROR (1) otherwise.
        """
        TTP_OK = 0
        TTP_ERROR = 1
        MASK_NORMAL = 0

        rank_mask = self._generate_rank_mask(rank_choose_info)
        rank_size = len(rank_mask)

        # Compute half consistency BEFORE the special case so both paths use it.
        # All dump ranks must come from the SAME half (either 0..ws/2-1 or ws/2..ws-1)
        # to produce a consistent optimizer state.
        first_half_err = False
        for error_rank in rank_choose_info['error_ranks']:
            if error_rank < self.world_size // 2:
                first_half_err = True
            if error_rank >= self.world_size // 2 and first_half_err:
                return TTP_ERROR

        if rank_size == rep_cnt:
            for rank, mask in rank_mask:
                if mask != MASK_NORMAL:
                    continue
                # Enforce same-half selection
                if first_half_err and rank < self.world_size // 2:
                    continue
                if not first_half_err and rank >= self.world_size // 2:
                    continue
                tmp_rank_vec.append(rank)
                return TTP_OK
            return TTP_ERROR

        offset = rank_size // rep_cnt
        for i in range(rep_cnt):
            rep_list = rank_choose_info['rank_vec'][i * offset : (i + 1) * offset]

            target_hit = all(
                self._is_rank_normal(rank, rank_mask)
                and (rank >= self.world_size // 2 if first_half_err else rank < self.world_size // 2)
                for rank in rep_list
            )

            if target_hit:
                tmp_rank_vec.extend(rep_list)
                return TTP_OK

        return TTP_ERROR

    def _generate_rank_mask(self, rank_choose_info: Dict) -> List[tuple]:
        """Generate rank mask list"""
        MASK_NORMAL = 0
        MASK_FAULT = 1

        rank_mask = []
        for rank in rank_choose_info['rank_vec']:
            if self.replica_manager:
                status = self.replica_manager.worker_status.get(rank)
                mask = MASK_FAULT if status in (WorkerStatus.FAULT, WorkerStatus.ABNORMAL) else MASK_NORMAL
            else:
                mask = MASK_NORMAL
            rank_mask.append((rank, mask))

        return rank_mask

    def _is_rank_normal(self, rank: int, rank_mask: List[tuple]) -> bool:
        """Check if rank is normal"""
        MASK_NORMAL = 0
        for r, mask in rank_mask:
            if r == rank:
                return mask == MASK_NORMAL
        return False

    def begin_exception_ckpt(self, error_ranks: Set[int]) -> int:
        """Build and send dump checkpoint requests to healthy replica ranks.

        Uses the replica group topology to select ranks that have complete data.
        Returns TTP_OK (0) on success, TTP_ERROR (1) if no healthy group is found.
        """
        TTP_OK = 0
        TTP_ERROR = 1

        current_iteration = 0
        with self._lock:
            # Prefer the fault rank's iteration — it just reported the error,
            # so its heartbeat value is the most accurate.  Arbitrary iteration
            # order of _worker_iteration (dict insertion order) otherwise causes
            # inconsistent global_step (sometimes 2, sometimes 3).
            for rank in error_ranks:
                if rank in self._worker_iteration and self._worker_iteration[rank] > 0:
                    current_iteration = self._worker_iteration[rank]
                    break
            # Fallback: no fault rank had a valid iteration, use any available
            if current_iteration == 0:
                for rank, iteration in self._worker_iteration.items():
                    if iteration > 0:
                        current_iteration = iteration
                        break

        send_group = {}
        send_world_group = set()
        idx = 0

        dp_group_list_map = self._get_dp_group_list_map()

        if not dp_group_list_map:
            return TTP_ERROR

        for rep_cnt, dp_groups in dp_group_list_map:
            for dp_group in dp_groups:
                tmp_rank_vec = []
                rank_choose_info = {'repair_step': current_iteration, 'error_ranks': error_ranks, 'rank_vec': dp_group}

                ret = self.choose_rank_inner_rl(rank_choose_info, tmp_rank_vec, rep_cnt)
                if ret != TTP_OK:
                    return ret

                for rank in tmp_rank_vec:
                    if rank not in send_group:
                        send_group[rank] = []
                    send_group[rank].append((idx, tmp_rank_vec))
                    send_world_group.add(rank)
            idx += 1

        with self._dump_lock:
            self._current_dump_ranks = list(send_world_group)

        for rank, rank_list in send_group.items():
            ckpt_msg = {
                'step': current_iteration,
                'num': len(rank_list),
                'sn': idx,
                'repair_id': self._current_request_id,
                'ranks': [],
                'world_ranks': list(send_world_group),
            }

            logger.error("[TTP] [BUILD_CKPT_MSG] rank=%s, rank_list=%s", rank, rank_list)

            for record in rank_list:
                ckpt_msg['ranks'].append(record[0])
                ckpt_msg['ranks'].append(len(record[1]))
                for rk in record[1]:
                    ckpt_msg['ranks'].append(rk)

            logger.error("[TTP] [BUILD_CKPT_MSG_DONE] rank=%s, ckpt_msg=%s", rank, ckpt_msg)

            self._send_dump_request_to_rank(rank, ckpt_msg)

        return TTP_OK

    def _get_dp_group_list_map(self) -> List[tuple]:
        """Get DP group list mapping"""
        if not self.replica_manager:
            return []

        dp_group_list_map = []

        if self.replica_manager.dp_cp_groups:
            rep_cnt = self.replica_manager.dp_cp_rep_cnt
            dp_group_list_map.append((rep_cnt, self.replica_manager.dp_cp_groups))

        if self.replica_manager.dp_ep_groups:
            rep_cnt = self.replica_manager.dp_ep_rep_cnt
            dp_group_list_map.append((rep_cnt, self.replica_manager.dp_ep_groups))

        return dp_group_list_map

    def _send_dump_request_to_rank(self, rank: int, ckpt_msg: Dict) -> bool:
        """Send dump request to specified rank"""
        request_data = json.dumps(ckpt_msg).encode('utf-8')
        return self.server.send_to_client(rank, MSG_TYPE_DUMP_REQUEST, request_data)

    def _send_dump_requests(self, dump_ranks: List[int]) -> bool:
        """Send dump requests"""
        success_count = 0

        with self._dump_lock:
            self._current_dump_ranks = dump_ranks
            request_id = self._current_request_id

        for rank in range(self.world_size):
            request_data = json.dumps(
                {
                    'request_id': request_id,
                    'fault_ranks': list(self._fault_ranks),
                    'dump_ranks': dump_ranks,
                    'timestamp': time.time(),
                }
            ).encode('utf-8')

            logger.debug("Sending dump request to rank %s", rank)
            success = self.server.send_to_client(rank, MSG_TYPE_DUMP_REQUEST, request_data)
            if success:
                success_count += 1
            else:
                logger.warning("Failed to send dump request to rank %s (connection may be lost)", rank)

        return success_count > 0

    def _on_trigger_dump(self, conn: ClientConnection, data: bytes) -> None:
        """Handle dump request triggered by main process"""
        try:
            request = json.loads(data.decode('utf-8'))
            fault_rank = request.get('fault_rank')

            logger.info("[DUMP_STEP_1] Received TRIGGER_DUMP from main process: fault_rank=%s", fault_rank)

            with self._lock:
                all_status = {rank: str(status) for rank, status in self._worker_status.items()}  # noqa: F841

            connected_ranks = self.server.get_connected_ranks()  # noqa: F841

            self._dump_complete_event.clear()
            self._dump_success = False

            if fault_rank is not None:
                self._on_worker_fault(fault_rank)
            else:
                fault_ranks = self._detect_fault_ranks()

                if fault_ranks:
                    for rank in fault_ranks:
                        self._on_worker_fault(rank)
                else:
                    dump_ranks = self._select_any_healthy_rank()

                    if dump_ranks:
                        with self._dump_lock:
                            self._dump_in_progress = True
                            self._fault_ranks.clear()

                        success = self._send_dump_requests(dump_ranks)

                        if not success:
                            logger.error("[DUMP_STEP_2B_2] Failed to send dump requests to any rank in %s", dump_ranks)
                            with self._dump_lock:
                                self._dump_in_progress = False
                                self._fault_ranks.clear()
                            response = json.dumps({'success': False, 'error': 'Failed to send dump requests'}).encode(
                                'utf-8'
                            )
                            conn.send_message(MSG_TYPE_TRIGGER_DUMP_ACK, response)
                            return
                    else:
                        response = json.dumps({'success': False, 'error': 'No healthy ranks'}).encode('utf-8')
                        conn.send_message(MSG_TYPE_TRIGGER_DUMP_ACK, response)
                        return

            success = self._dump_complete_event.wait(timeout=300.0)

            response = json.dumps(
                {
                    'success': success,
                    'iteration': self._dump_iteration,
                }
            ).encode('utf-8')
            conn.send_message(MSG_TYPE_TRIGGER_DUMP_ACK, response)

        except Exception as e:
            logger.error("[DUMP_ERROR] Failed to handle TRIGGER_DUMP: %s", e, exc_info=True)
            response = json.dumps({'success': False, 'error': str(e)}).encode('utf-8')
            conn.send_message(MSG_TYPE_TRIGGER_DUMP_ACK, response)

    def _detect_fault_ranks(self) -> List[int]:
        """Detect fault rank list"""
        fault_ranks = []
        with self._lock:
            for rank, status in self._worker_status.items():
                if status in [WorkerStatus.FAULT, WorkerStatus.ABNORMAL]:
                    fault_ranks.append(rank)
        return fault_ranks

    def _select_any_healthy_rank(self) -> List[int]:
        """Select any healthy rank for dump"""
        with self._lock:
            for rank, status in self._worker_status.items():
                if status == WorkerStatus.NORMAL:
                    return [rank]
        return []


def main():
    """Controller standalone process entry point"""

    config = get_ttp_config()
    world_size = int(os.environ.get('TTP_WORLD_SIZE', '1'))

    controller = TTPController(rank=0, world_size=world_size, config=config)
    controller.start(config.server_ip, config.server_port)
    TTPController.set_instance(controller)

    def handle_shutdown(signum, frame):
        logger.info("Received signal %s, waiting for dump to complete...", signum)

        if controller._dump_in_progress:
            controller._dump_complete_event.wait(timeout=300.0)

        controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    logger.info("Controller process running...")

    last_activity_time = time.time()
    idle_timeout = 600

    try:
        while controller.running:
            time.sleep(1)

            with controller._lock:
                active_workers = len([s for s in controller._worker_status.values() if s != WorkerStatus.FAULT])

            if active_workers > 0:
                last_activity_time = time.time()
            elif time.time() - last_activity_time > idle_timeout:
                logger.info("No active workers for %ss, exiting...", idle_timeout)
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    finally:
        controller.stop()


if __name__ == '__main__':
    main()
