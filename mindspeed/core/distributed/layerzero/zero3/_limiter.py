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
