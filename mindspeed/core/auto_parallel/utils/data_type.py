# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
from enum import Enum


# profilling task type
class TaskType(Enum):
    EXIT_SEARCH = -1
    OPERATOR_PROFILLING = 1


# search algorithm
class SearchAlgo(Enum):
    FULL_PRECISION = 'full_precision'
    FAST_MODE = 'fast_mode'