# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import socket

HEADER_SIZE = 8
HEADER_FORMAT = '!hhI'


def recv_exact(sock, size: int):
    """Read exactly `size` bytes from the socket"""
    data = b''
    while len(data) < size:
        try:
            chunk = sock.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        except socket.error:
            return None
    return data
