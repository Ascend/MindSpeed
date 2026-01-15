# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import os
from logging.config import dictConfig
from typing import List, Callable

import torch

_warned_messages = set()


def set_log_level(level="INFO"):
    """
    level: INFO, DEBUG, WARNING, ERROR, CRITICAL
    """
    rank = os.getenv("RANK", 0)
    local_rank = os.getenv("LOCAL_RANK", 0)
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": f"[Rank {rank} | Local Rank {local_rank}] %(asctime)s "
                          "%(levelname)s [%(name)s:%(lineno)d] => %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": f"{level}",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": f"{level}",
        },
    }
    dictConfig(config)


def print_rank(log: Callable, message: str, ranks: [int, List[int]] = 0):
    if torch.distributed.is_initialized():
        if isinstance(ranks, int):
            ranks = [ranks]
        if torch.distributed.get_rank() in ranks:
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
