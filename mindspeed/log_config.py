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
