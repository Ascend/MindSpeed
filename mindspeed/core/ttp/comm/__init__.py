# Copyright 2025 Huawei Technologies Co., Ltd.
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
