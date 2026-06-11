# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import socket
import struct
import threading
import time
from typing import Callable, Dict, Optional, Set

from ..constants import MSG_TYPE_REGISTER
from . import HEADER_SIZE, HEADER_FORMAT, recv_exact

logger = logging.getLogger(__name__)


class ClientConnection:
    """Client connection"""

    def __init__(self, client_socket: socket.socket, client_addr: tuple):
        self.socket = client_socket
        self.addr = client_addr
        self.rank: Optional[int] = None
        self.last_heartbeat_time = time.time()
        self.lock = threading.Lock()

    def send_message(self, msg_type: int, data: bytes = b'') -> bool:
        """Send a message"""
        with self.lock:
            try:
                header = struct.pack(HEADER_FORMAT, msg_type, 0, len(data))
                self.socket.sendall(header + data)
                return True
            except socket.error:
                return False

    def recv_message(self, timeout: float = 1.0) -> Optional[tuple]:
        """Receive a message"""
        try:
            self.socket.settimeout(timeout)
            header = recv_exact(self.socket, HEADER_SIZE)
            if not header:
                return None

            msg_type, result, data_len = struct.unpack(HEADER_FORMAT, header)
            data = recv_exact(self.socket, data_len) if data_len > 0 else b''
            return msg_type, result, data
        except socket.error:
            return None

    def close(self) -> None:
        """Close the connection"""
        try:
            self.socket.close()
        except Exception:
            logger.warning("Failed to close connection socket", exc_info=True)

    def update_heartbeat(self) -> None:
        """Update heartbeat timestamp"""
        self.last_heartbeat_time = time.time()


class SocketServer:
    """TCP Socket server"""

    def __init__(self, host: str = '0.0.0.0', port: int = 29500):
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.accept_thread: Optional[threading.Thread] = None
        self.receive_threads: Dict[int, threading.Thread] = {}
        self.client_connections: Dict[int, ClientConnection] = {}
        self.connections_lock = threading.Lock()
        self.handlers: Dict[int, Callable] = {}
        self.disconnect_handler: Optional[Callable] = None

    def start(self) -> bool:
        """Start the server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(128)
            self.running = True
            self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
            self.accept_thread.start()
            return True
        except socket.error as e:
            raise RuntimeError(f"Failed to start server on {self.host}:{self.port}: {e}")

    def stop(self) -> None:
        """Stop the server"""
        self.running = False
        with self.connections_lock:
            for conn in self.client_connections.values():
                conn.close()
            self.client_connections.clear()

        threads_to_join = list(self.receive_threads.values())
        for thread in threads_to_join:
            thread.join(timeout=1.0)
        self.receive_threads.clear()

        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                logger.warning("Failed to close server socket", exc_info=True)
            self.server_socket = None

    def register_handler(self, msg_type: int, handler: Callable) -> None:
        """Register a message handler"""
        self.handlers[msg_type] = handler

    def register_disconnect_handler(self, handler: Callable) -> None:
        """Register a disconnect handler"""
        self.disconnect_handler = handler

    def send_to_client(self, rank: int, msg_type: int, data: bytes = b'') -> bool:
        """Send a message to a specific client"""
        with self.connections_lock:
            conn = self.client_connections.get(rank)
            if conn:
                return conn.send_message(msg_type, data)
        return False

    def get_connected_ranks(self) -> Set[int]:
        """Get connected ranks (only registered integer ranks, excluding temp_ keys)"""
        with self.connections_lock:
            return {k for k in self.client_connections if isinstance(k, int)}

    def _accept_loop(self) -> None:
        """Accept connection loop"""
        while self.running and self.server_socket:
            try:
                self.server_socket.settimeout(1.0)
                try:
                    client_socket, client_addr = self.server_socket.accept()
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    conn = ClientConnection(client_socket, client_addr)

                    temp_key = f"temp_{id(conn)}"
                    with self.connections_lock:
                        self.client_connections[temp_key] = conn

                    receive_thread = threading.Thread(
                        target=self._receive_loop_for_new_connection, args=(conn, temp_key), daemon=True
                    )
                    receive_thread.start()

                except socket.timeout:
                    continue
            except Exception:
                break

    def _receive_loop_for_new_connection(self, conn: ClientConnection, temp_key: str) -> None:
        """Receive loop for new connections (waiting for registration)"""
        first_message_received = False

        while self.running and not first_message_received:
            try:
                result = conn.recv_message(timeout=1.0)
                if result is None:
                    continue

                msg_type, msg_result, data = result

                if msg_type == MSG_TYPE_REGISTER:
                    import json

                    try:
                        request = json.loads(data.decode('utf-8'))
                        rank = request.get('rank')
                        if rank is not None:
                            conn.rank = rank

                            with self.connections_lock:
                                if temp_key in self.client_connections:
                                    del self.client_connections[temp_key]
                                self.client_connections[rank] = conn

                            handler = self.handlers.get(msg_type)
                            if handler:
                                handler(conn, data)

                            receive_thread = threading.Thread(target=self._receive_loop, args=(conn,), daemon=True)
                            receive_thread.start()
                            self.receive_threads[rank] = receive_thread

                            first_message_received = True
                    except Exception:
                        logger.warning("Failed to accept connection", exc_info=True)
            except Exception:
                break

        if not first_message_received:
            with self.connections_lock:
                self.client_connections.pop(temp_key, None)
            conn.close()

    def _receive_loop(self, conn: ClientConnection) -> None:
        """Receive message loop"""
        while self.running and conn.rank is not None:
            try:
                result = conn.recv_message(timeout=1.0)
                if result is None:
                    continue

                msg_type, msg_result, data = result
                handler = self.handlers.get(msg_type)
                if handler:
                    handler(conn, data)
            except Exception:
                break

        if conn.rank is not None:
            self.remove_connection(conn.rank)

    def remove_connection(self, rank: int) -> None:
        """Remove a client connection"""
        with self.connections_lock:
            conn = self.client_connections.pop(rank, None)
            if conn:
                conn.close()

        self.receive_threads.pop(rank, None)

        if self.disconnect_handler:
            self.disconnect_handler(rank)
