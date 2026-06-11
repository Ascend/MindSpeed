# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import socket
import struct
import threading
import time
from typing import Callable, Dict, Optional, Tuple

from . import HEADER_SIZE, HEADER_FORMAT, recv_exact

logger = logging.getLogger(__name__)


class SocketClient:
    """TCP Socket client"""

    def __init__(self, server_ip: str, server_port: int):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.lock = threading.Lock()
        self.handlers: Dict[int, Callable] = {}
        self.receive_thread: Optional[threading.Thread] = None
        self.running = False

        self._response_condition = threading.Condition()
        self._response_data = None
        self._response_received = False
        self._expected_sn = None
        self._sn_counter = 0
        self._sn_lock = threading.Lock()

    def connect(self, max_retry_times: int = 5, retry_interval: float = 1.0) -> bool:
        """Connect to the server"""
        for attempt in range(max_retry_times):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.socket.connect((self.server_ip, self.server_port))
                self.connected = True
                self.running = True
                return True
            except socket.error as e:
                if self.socket:
                    self.socket.close()
                    self.socket = None
                if attempt < max_retry_times - 1:
                    time.sleep(retry_interval)
                else:
                    raise ConnectionError(f"Failed to connect to {self.server_ip}:{self.server_port}: {e}")
        return False

    def _generate_sn(self) -> int:
        """Generate a sequence number"""
        with self._sn_lock:
            self._sn_counter += 1
            return self._sn_counter

    def start_receive_loop(self) -> None:
        """Start the receive loop"""
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()

    def disconnect(self) -> None:
        """Disconnect"""
        self.running = False
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                logger.warning("Failed to close client socket", exc_info=True)
            self.socket = None

    def register_handler(self, msg_type: int, handler: Callable) -> None:
        """Register a message handler"""
        self.handlers[msg_type] = handler

    def send_message(self, msg_type: int, data: bytes = b'') -> bool:
        """Send a message"""
        with self.lock:
            if not self.connected or not self.socket:
                return False

            try:
                header = struct.pack(HEADER_FORMAT, msg_type, 0, len(data))
                self.socket.sendall(header + data)
                return True
            except socket.error:
                self.connected = False
                return False

    def send_message_with_response(self, msg_type: int, data: bytes = b'', timeout: float = 30.0) -> Tuple[bool, bytes]:
        """Send a message and wait for response"""
        import json

        sn = self._generate_sn()

        with self._response_condition:
            self._response_received = False
            self._response_data = None
            self._expected_sn = sn

            try:
                message = json.loads(data.decode('utf-8')) if data else {}
                message['sn'] = sn
                request_data = json.dumps(message).encode('utf-8')

                with self.lock:
                    if not self.connected or not self.socket:
                        return False, b''
                    header = struct.pack(HEADER_FORMAT, msg_type, 0, len(request_data))
                    self.socket.sendall(header + request_data)

                start_time = time.time()
                while not self._response_received:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        return False, b''

                    self._response_condition.wait(remaining)

                resp_msg_type, resp_result, resp_data = self._response_data
                return resp_result == 0, resp_data

            finally:
                self._expected_sn = None
                self._response_received = False
                self._response_data = None

    def _process_one_message(self) -> bool:
        """Try to receive and dispatch one message. Returns False on error."""
        import json

        if not self.socket:
            return False
        self.socket.settimeout(0.1)
        try:
            header = recv_exact(self.socket, HEADER_SIZE)
            if header:
                msg_type, result, data_len = struct.unpack(HEADER_FORMAT, header)
                data = recv_exact(self.socket, data_len) if data_len > 0 else b''

                message = json.loads(data.decode('utf-8')) if data else {}
                sn = message.get('sn')

                if sn is not None and sn == self._expected_sn:
                    with self._response_condition:
                        self._response_data = (msg_type, result, data)
                        self._response_received = True
                        self._response_condition.notify()
                else:
                    handler = self.handlers.get(msg_type)
                    if handler:
                        handler(data)
        except socket.timeout:
            return False
        except Exception:
            time.sleep(0.1)
            return False
        return True

    def _receive_loop(self) -> None:
        """Receive message loop"""
        while self.running and self.connected:
            try:
                self._process_one_message()
            except Exception:
                time.sleep(0.1)
                continue
