# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import Dict, Tuple, TYPE_CHECKING, Optional, Deque
from collections import deque
from abc import abstractmethod
from contextlib import contextmanager

import torch
    
    
class EventQueueBase:
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    @contextmanager
    def block(self):
        ...
        
    @abstractmethod
    def empty(self):
        ...
        
    @abstractmethod
    def enqueue(self, free_event: torch.cuda.Event) -> None:
        ...
        
    @abstractmethod
    def pop_left(self) -> Optional[torch.cuda.Event]:
        ...


class CriticalPathEventQueue(EventQueueBase):
    
    def __init__(self):
        super().__init__()
        self._queue: Deque[torch.cuda.Event] = deque()
        self._buffer: Deque[torch.cuda.Event] = deque()
        self.__blocked = False
    
    @contextmanager
    def block(self):
        try:
            self.__blocked = True
            yield
        finally:
            for event in self._buffer:
                self.enqueue(event)
            self._buffer.clear()
            self.__blocked = False
            
        
    def empty(self):
        return len(self._queue) == 0
    
    def enqueue(self, free_event: torch.cuda.Event) -> None:
        if self.__blocked:
            self._buffer.append(free_event)
        else:
            self._queue.append(free_event)
        
    @abstractmethod
    def pop_left(self) -> Optional[torch.cuda.Event]:
        if self._queue:
            event = self._queue.popleft()
            return event
        return None
        