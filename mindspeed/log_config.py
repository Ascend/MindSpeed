# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Configuration of logging."""

import os
from logging.config import dictConfig
from logging import Logger
from typing import Callable

import torch

LOG_LEVEL = os.getenv("MIND_SPEED_LOG_LEVEL", "INFO")
RANK = os.getenv("RANK", 0)
LOCAL_RANK = os.getenv("LOCAL_RANK", 0)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": f"[Rank {RANK} | Local Rank {LOCAL_RANK}] %(asctime)s "
            "%(levelname)s [%(name)s:%(lineno)d] => %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": f"{LOG_LEVEL}",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": f"{LOG_LEVEL}",
    },
}

_warned_messages = set()


def set_log_config():
    """Make log config effect."""
    dictConfig(LOGGING_CONFIG)


def log_rank_0(log: Callable, message: str):
    """If distributed is initialized, Log only in rank 0.

    Args:
        log (Logger): A function which can log message.
            such as:
            ```python
                LOG = getLogger(__name__)
                log_rank_0(LOG.INFO, "message")
            ```
        message (str): The log message.
    """
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            log(message)
    else:
        log(message)


def log_warning_once(logger, message):
    """
    Logs a warning message only once. Subsequent calls with the same message
    will be ignored.
    """
    if message not in _warned_messages:
        logger.warning(message)
        _warned_messages.add(message)