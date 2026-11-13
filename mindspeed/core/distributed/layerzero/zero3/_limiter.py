# Copyright 2024 Huawei Technologies Co., Ltd.
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

import collections


class _FreeEventQueue:

    def __init__(self, num_inflights: int = 3) -> None:
        self._queue = collections.deque()
        self._max_num_inflight_all_gathers = num_inflights

    def enqueue(self, free_event) -> None:
        """Enqueues a free event."""
        self._queue.append(free_event)

    def dequeue_if_needed(self):
        """Dequeues a single event if the limit is reached."""
        if len(self._queue) >= self._max_num_inflight_all_gathers:
            return self._dequeue()
        return None

    def _dequeue(self):
        """Dequeues a free event if possible."""
        if self._queue:
            event = self._queue.popleft()
            return event
        return None
