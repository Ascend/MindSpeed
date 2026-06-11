# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import json
import logging
import threading
import time
from typing import Callable, Optional

from ..config import HeartbeatConfig
from ..constants import MSG_TYPE_HEARTBEAT, WorkerStatus

logger = logging.getLogger(__name__)


class HeartbeatManager:
    """Heartbeat manager (worker side)"""

    STOP_TIMEOUT = 2.0

    def __init__(
        self,
        config: HeartbeatConfig,
        send_func: Callable[[int, bytes], bool],
        on_timeout: Optional[Callable[[int], None]] = None,
    ):
        self.config = config
        self.send_func = send_func
        self.on_timeout = on_timeout

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._current_iteration: int = 0
        self._current_status: WorkerStatus = WorkerStatus.NORMAL
        self._missed_count: int = 0
        self._last_heartbeat_time: float = 0.0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

    @property
    def running(self) -> bool:
        """Whether the manager is running"""
        return self._running and not self._stop_event.is_set()

    @property
    def missed_count(self) -> int:
        """Consecutive missed heartbeat count"""
        with self._lock:
            return self._missed_count

    @property
    def last_heartbeat_time(self) -> float:
        """Last heartbeat time"""
        with self._lock:
            return self._last_heartbeat_time

    def start(self) -> None:
        """Start heartbeat sending"""
        logger.debug("Starting HeartbeatManager")
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        logger.debug("HeartbeatManager started")

    def stop(self) -> None:
        """Stop heartbeat (graceful shutdown)"""
        logger.debug("Stopping HeartbeatManager")
        self._stop_event.set()
        self._running = False

        if self._thread:
            self._thread.join(timeout=self.STOP_TIMEOUT)
            if self._thread.is_alive():
                logger.warning("HeartbeatManager thread did not stop gracefully")

        logger.debug("HeartbeatManager stopped")

    def update_status(self, status: WorkerStatus, iteration: int = None) -> None:
        """Update status (thread-safe)"""
        with self._lock:
            self._current_status = status
            if iteration is not None:
                self._current_iteration = iteration

    def update_iteration(self, iteration: int) -> None:
        """Update iteration count (thread-safe)"""
        with self._lock:
            self._current_iteration = iteration

    def _heartbeat_loop(self) -> None:
        """Heartbeat send loop"""
        while not self._stop_event.is_set():
            try:
                self._send_heartbeat()
                interval = self.config.interval_ms / 1000.0
                if self._stop_event.wait(interval):
                    break
            except Exception as e:
                logger.error("Error in heartbeat loop: %s", e, exc_info=True)

        logger.debug("Heartbeat loop exited")

    def _send_heartbeat(self) -> None:
        """Send heartbeat"""
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

        success = self.send_func(MSG_TYPE_HEARTBEAT, heartbeat_data)

        with self._lock:
            if not success:
                self._missed_count += 1
                logger.warning("Heartbeat send failed, missed count: %s", self._missed_count)
                if self._missed_count >= self.config.max_missed_count:
                    logger.error("Heartbeat timeout reached (missed_count=%s)", self._missed_count)
                    if self.on_timeout:
                        try:
                            self.on_timeout(self._missed_count)
                        except Exception as e:
                            logger.error("Error in heartbeat timeout callback: %s", e, exc_info=True)
            else:
                self._missed_count = 0
                self._last_heartbeat_time = time.time()


class HeartbeatChecker:
    """Heartbeat checker (server side)"""

    STOP_TIMEOUT = 2.0
    INIT_TIMEOUT_SECONDS = 120.0

    def __init__(self, config: HeartbeatConfig, on_timeout: Callable[[int], None], world_size: int = 0):
        self.config = config
        self.on_timeout = on_timeout
        self.world_size = world_size

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._worker_last_heartbeat: dict = {}
        self._worker_missed_count: dict = {}
        self._worker_status: dict = {}
        self._workers_ready: set = set()
        self._first_register_time: float = 0.0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

    @property
    def running(self) -> bool:
        """Whether the checker is running"""
        return self._running and not self._stop_event.is_set()

    def start(self) -> None:
        """Start checking"""
        logger.debug("Starting HeartbeatChecker")
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()
        logger.debug("HeartbeatChecker started")

    def stop(self) -> None:
        """Stop checking (graceful shutdown)"""
        logger.debug("Stopping HeartbeatChecker")
        self._stop_event.set()
        self._running = False

        if self._thread:
            self._thread.join(timeout=self.STOP_TIMEOUT)
            if self._thread.is_alive():
                logger.warning("HeartbeatChecker thread did not stop gracefully")

        logger.debug("HeartbeatChecker stopped")

    def register_worker(self, rank: int) -> None:
        """Register worker"""
        with self._lock:
            self._worker_last_heartbeat[rank] = time.time()
            self._worker_missed_count[rank] = 0
            self._worker_status[rank] = WorkerStatus.NORMAL
            if self._first_register_time == 0.0:
                self._first_register_time = time.time()
        logger.debug("Worker rank %s registered for heartbeat checking", rank)

    def unregister_worker(self, rank: int) -> None:
        """Unregister worker"""
        with self._lock:
            self._worker_last_heartbeat.pop(rank, None)
            self._worker_missed_count.pop(rank, None)
            self._worker_status.pop(rank, None)
        logger.debug("Worker rank %s unregistered from heartbeat checking", rank)

    def update_heartbeat(self, rank: int, status: WorkerStatus = WorkerStatus.NORMAL) -> None:
        """Update heartbeat"""
        with self._lock:
            self._worker_last_heartbeat[rank] = time.time()
            self._worker_missed_count[rank] = 0
            self._worker_status[rank] = status

            if rank not in self._workers_ready:
                self._workers_ready.add(rank)

    def get_worker_status(self, rank: int) -> WorkerStatus:
        """Get worker status"""
        with self._lock:
            return self._worker_status.get(rank, WorkerStatus.FAULT)

    def _check_loop(self) -> None:
        """Check loop"""
        while not self._stop_event.is_set():
            try:
                self._check_timeouts()
                interval = self.config.timeout_ms / 1000.0

                if self._stop_event.wait(interval):
                    break
            except Exception as e:
                logger.error("Error in heartbeat check loop: %s", e, exc_info=True)

        logger.debug("Heartbeat check loop exited")

    def _check_timeouts(self) -> None:
        """Check timeouts"""
        current_time = time.time()
        timeout_threshold = self.config.timeout_ms / 1000.0

        with self._lock:
            ready_count = len(self._workers_ready)
            first_register_time = self._first_register_time

        if ready_count < self.world_size:
            if first_register_time == 0.0 or current_time - first_register_time < self.INIT_TIMEOUT_SECONDS:
                return

        with self._lock:
            for rank, last_time in list(self._worker_last_heartbeat.items()):
                worker_status = self._worker_status.get(rank, WorkerStatus.NORMAL)
                if worker_status in (WorkerStatus.ABNORMAL, WorkerStatus.PAUSE):
                    continue

                if current_time - last_time > timeout_threshold:
                    self._worker_missed_count[rank] = self._worker_missed_count.get(rank, 0) + 1
                    missed = self._worker_missed_count[rank]

                    if missed >= self.config.max_missed_count:
                        self._worker_status[rank] = WorkerStatus.FAULT
                        try:
                            self.on_timeout(rank)
                        except Exception as e:
                            logger.error("Error in worker timeout callback for rank %s: %s", rank, e, exc_info=True)
                else:
                    self._worker_missed_count[rank] = 0
